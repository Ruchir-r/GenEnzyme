"""
Unified top-k beam inference for both P07 and RAG model families.

Produces N candidate predictions per test row, writes a CSV with N rows
per input (one per beam rank). Used to measure whether the model's
correct recombination prediction is in the top-N distribution even
when greedy decoding misses it.

Auto-detects the architecture from --model-type:
  p07 : GVPIdAwareMHGGAT  (fair-P07, p07-simsplit, p07-original)
  rag : GVPRAGMHGGAT      (RAG-fair, RAG-fairwarm)

Examples:
  # Fair-P07 top-5 beam at sim30
  python3 infer_topk_all.py --model-type p07 \\
      --ckpt    .../15_p07_fair/checkpoints/p07_fair_simsplit30_seed0/best_id_curriculum_model.pt \\
      --test-csv .../09_benchmarks/results/lencheck_p07_simsplit30_test.clean.csv \\
      --out-csv  inference_detailed_simsplit30_p07_fair_beam5.csv \\
      --num-candidates 5

  # RAG-fairwarm top-5 beam at sim40 (requires --retrieval-dir)
  python3 infer_topk_all.py --model-type rag \\
      --ckpt    .../15_p07_fair/checkpoints/rag_fairwarm_simsplit40_seed0/best_p11_model.pt \\
      --test-csv .../09_benchmarks/results/lencheck_p07_simsplit40_test.clean.csv \\
      --retrieval-dir .../data/foldseek_cache_simsplit40 \\
      --out-csv  inference_detailed_simsplit40_rag_fairwarm_beam5.csv \\
      --num-candidates 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdChemReactions
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ---- import paths --------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_LIB_DIR  = _THIS_DIR.parent / "_lib"
_P07_DIR  = _THIS_DIR.parent / "07_id_curriculum"
_P11_DIR  = _THIS_DIR.parent / "11_foldseek_rag"
for p in (_LIB_DIR, _P07_DIR, _P11_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---- Reaction-Tanimoto scoring -------------------------------------------
# Uses rdChemReactions structural fingerprints — matches the legacy
# infer_3way.py:rxnpfp_similarity() function. Plain Chem.MolFromSmiles()
# returns None on reaction strings (the `>>` is not valid SMILES), so the
# earlier per-molecule scoring silently failed on every row.
def _heal_rxn(smi: str) -> str:
    """Fix obvious reaction-SMILES syntax errors (mirror of _lib/infer_3way.heal_reaction_smiles)."""
    if not isinstance(smi, str) or not smi:
        return ""
    if smi.count(">>") > 1:
        parts = smi.split(">>")
        smi = f"{parts[0]}>>{parts[-1]}"
    elif ">>" not in smi:
        smi = f">>{smi}"
    return smi.strip(".")


def tanimoto(true_smi: str, pred_smi: str) -> tuple[float, bool]:
    if not isinstance(true_smi, str) or not isinstance(pred_smi, str):
        return 0.0, False
    if not true_smi or not pred_smi:
        return 0.0, False
    try:
        r1 = rdChemReactions.ReactionFromSmarts(true_smi, useSmiles=True)
        r2 = rdChemReactions.ReactionFromSmarts(_heal_rxn(pred_smi), useSmiles=True)
        if r1 is None or r2 is None:
            return 0.0, False
        f1 = rdChemReactions.CreateStructuralFingerprintForReaction(r1)
        f2 = rdChemReactions.CreateStructuralFingerprintForReaction(r2)
        return float(DataStructs.TanimotoSimilarity(f1, f2)), True
    except Exception:
        return 0.0, False


def _decode_one(seq_ids: torch.Tensor, inv_vocab: dict[int, str],
                START_ID: int, END_ID: int, PAD_ID: int) -> str:
    tokens = []
    for tid in seq_ids.tolist():
        if tid == END_ID:
            break
        if tid in (START_ID, PAD_ID):
            continue
        tok = inv_vocab.get(tid, "")
        if tok and tok not in ("<START>", "<PAD>", "<END>"):
            tokens.append(tok)
    return "".join(tokens)


# ==========================================================================
D17_PATH = "/path/to/struct_rxnpred/claude_dir/17_reaction_revival"
D17_COMP_CACHE = "/path/to/struct_rxnpred/claude_dir/17_reaction_revival/comp_table_em_ext.json"


def _compose(*procs):
    """Chain logits processors left to right; None entries are skipped."""
    live = [p for p in procs if p is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def _chained(logits, ids):
        for p in live:
            logits = p(logits, ids)
        return logits
    return _chained


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-type", choices=("p07", "rag"), required=True,
                    help="p07: GVPIdAwareMHGGAT. rag: GVPRAGMHGGAT (also needs --retrieval-dir).")
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--test-csv",   required=True)
    ap.add_argument("--vocab-path", default=None,
                    help="Default: read from config.")
    ap.add_argument("--embeds-dir", default=None,
                    help="Default: read from config.")
    ap.add_argument("--retrieval-dir", default=None,
                    help="Required for --model-type rag.")
    ap.add_argument("--top-k-rag",            type=int, default=5,
                    help="RAG top-k retrieved enzymes (matches model training).")
    ap.add_argument("--max-retrieved-tokens", type=int, default=64,
                    help="RAG max tokens per retrieved reaction.")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", choices=("beam", "sample", "hybrid"), default="beam")
    ap.add_argument("--beam-frac", type=float, default=0.0625,
                    help="mode=hybrid: fraction of the candidate budget spent on "
                         "beam, remainder sampled. Keep small: literature-optimal "
                         "beam widths are 30-50 (arXiv 1706.03872) and our "
                         "matched-budget test had beam25 at 7.17 pct coverage vs "
                         "sample25 at 10.28 pct.")
    ap.add_argument("--constrain", choices=("none", "grammar", "composition"),
                    default="none",
                    help="grammar: single arrow, no empty components, per-side "
                         "multiplicity cap, END only after a product metabolite "
                         "-- all satisfied by every em_ext training reaction. "
                         "composition: also gate product tokens on heavy-atom "
                         "reachability vs the GENERATED reactant side "
                         "(MARLIN-style). composition is NOT provably lossless "
                         "here because reactants are generated, not given.")
    ap.add_argument("--mult-cap", type=int, default=4,
                    help="Max repeats of one metabolite per side. 4 = measured "
                         "max in em_ext train, so it forbids nothing real.")
    ap.add_argument("--comp-cache", default=D17_COMP_CACHE,
                    help="Where to cache the per-token heavy-atom table.")
    ap.add_argument("--num-candidates", type=int, default=5,
                    help="Beam width (mode=beam) or num samples per input (mode=sample).")
    ap.add_argument("--top-k",       type=int, default=20,
                    help="For mode=sample: keep top-k logits before sampling.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.0,
                    help="Nucleus sampling: keep the smallest set of tokens whose cumulative "
                         "probability >= top-p, mask the rest. 0 disables (default). "
                         "Applied before --top-k; see --top-k note.")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--length-norm", type=float, default=1.0,
                    help="Beam-search length penalty exponent. 0=no normalization "
                         "(biases toward short hypotheses); 1.0=full normalization "
                         "(balances long vs short; recommended default).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Smaller than greedy: beam_width inflates mem ~Nx.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="If set, run only this many rows (random subset, seed 0).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--id-override", action="store_true", default=True,
                    help="Apply identity-reaction override (on by default for P07/RAG).")
    ap.add_argument("--no-id-override", dest="id_override", action="store_false")
    ap.add_argument("--id-override-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed torch/cuda RNG and enable deterministic algorithms. "
                         "Sampling mode is random without this; beam is near-deterministic "
                         "but reduction order can still flip near-ties.")
    ap.add_argument("--fp32", action="store_true",
                    help="Disable bfloat16 autocast; run the decoder in fp32. "
                         "bf16 has ~3 decimal digits, enough to flip near-tied beams.")
    args = ap.parse_args()
    # --- determinism -------------------------------------------------------
    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:
            print("[seed] use_deterministic_algorithms unavailable:", _e)
        print(f"[seed] seed={args.seed} deterministic=on tf32=off")
    print(f"[seed] autocast_bf16={'off (fp32)' if args.fp32 else 'on'}")


    if args.model_type == "rag" and not args.retrieval_dir:
        ap.error("--model-type rag requires --retrieval-dir.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- imports that depend on sys.path setup ---
    from config import SPECIAL_TOKENS, DATA_CONFIG, TRAIN_CONFIG
    vocab_path = args.vocab_path or DATA_CONFIG["vocab_path"]
    embeds_dir = args.embeds_dir or DATA_CONFIG["embeddings_dir"]

    print(f"[topk-all] device={device}  arch={args.model_type}  mode={args.mode}  N={args.num_candidates}")
    print(f"[topk-all] ckpt={args.ckpt}")
    print(f"[topk-all] test={args.test_csv}")
    print(f"[topk-all] out={args.out_csv}")
    if args.model_type == "rag":
        print(f"[topk-all] retrieval_dir={args.retrieval_dir}  rag_top_k={args.top_k_rag}  max_R={args.max_retrieved_tokens}")

    with open(vocab_path) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    # models are built from TRAIN_CONFIG["vocab_size"] (default 8882); override to
    # the actual vocab so checkpoints trained on a custom vocab load correctly.
    TRAIN_CONFIG["vocab_size"] = max(vocab.values()) + 1
    START_ID = SPECIAL_TOKENS["<START>"]
    END_ID   = SPECIAL_TOKENS["<END>"]
    PAD_ID   = SPECIAL_TOKENS["<PAD>"]
    ignore = {PAD_ID, START_ID, END_ID}

    # --- dataset + model construction (per architecture) ---
    if args.model_type == "p07":
        from data_curriculum import CurriculumGraphReactionDataset
        from mhggat_id_aware import GVPIdAwareMHGGAT, decode_with_identity_override
        ds = CurriculumGraphReactionDataset(args.test_csv, vocab_path, embeds_dir)
        # Structure-encoder geometry must match the checkpoint. Training takes
        # these as CLI flags (--num-mpnn-layers/--d-v/--num-rbf); inference used
        # to hardcode the constructor defaults, so any checkpoint trained with
        # d_v != 16 or num_mpnn_layers != 2 died with a state_dict size mismatch.
        # The env defaults below reproduce the previous behaviour exactly, so
        # existing callers are unaffected.
        model = GVPIdAwareMHGGAT(
            num_mpnn_layers=int(os.environ.get("RXN_NUM_MPNN_LAYERS", 2)),
            d_v=int(os.environ.get("RXN_D_V", 16)),
            num_rbf=int(os.environ.get("RXN_NUM_RBF", 16)),
        ).to(device)
    else:  # rag
        from data_rag import RAGDataset
        from mhggat_rag import GVPRAGMHGGAT
        # RAG model has decode_with_identity_override at the same import level as P07 normally;
        # try both locations.
        try:
            from mhggat_id_aware import decode_with_identity_override
        except ImportError:
            from mhggat_rag import decode_with_identity_override
        ds = RAGDataset(args.test_csv, vocab_path, embeds_dir,
                        retrieval_dir=args.retrieval_dir,
                        top_k=args.top_k_rag,
                        max_retrieved_tokens=args.max_retrieved_tokens)
        model = GVPRAGMHGGAT(top_k=args.top_k_rag,
                             max_retrieved_tokens=args.max_retrieved_tokens).to(device)

    # subset
    if args.n_samples is not None and args.n_samples < len(ds):
        rng = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(ds), generator=rng)[:args.n_samples].tolist()
        sub = Subset(ds, idx)
    else:
        sub = ds
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        collate_fn=ds.collate_fn, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    print(f"[topk-all] rows: {len(sub)}  batches: {len(loader)}")

    # load ckpt
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[topk-all] ckpt loaded — missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()

    # --- inference loop ---
    # ---- nucleus (top-p) processor ---------------------------------------
    _nucleus = None
    if args.top_p and args.top_p > 0.0:
        if not (0.0 < args.top_p <= 1.0):
            raise SystemExit("--top-p must be in (0, 1]")
        if args.mode == "beam":
            raise SystemExit("--top-p is meaningless for mode=beam (no sampling step)")
        # top-k runs AFTER this mask inside top_k_batch_sampling, so a small top-k would
        # silently truncate the nucleus. Widen it unless the caller chose a value explicitly.
        if args.top_k == 20:
            # Widen enough that the nucleus is effectively untruncated, but NOT to the whole
            # 16,053-token vocabulary: job 641166 OOM'd at 74.2 GiB because sorting/softmaxing
            # the full vocab across 400 candidate rows blows up the activation footprint. At
            # p<=0.95 the nucleus on this model is far smaller than 512 tokens, so this bound is
            # not binding in practice -- and if it ever were, --top-k can be set explicitly.
            args.top_k = 512
            print("[nucleus] --top-k left at default; widened to 512 (bounded to avoid the "
                  "full-vocab OOM seen in job 641166)", flush=True)

        def _nucleus_proc(logits, _ids, _p=args.top_p):
            srt, idx = torch.sort(logits, dim=-1, descending=True)
            cum = torch.softmax(srt, dim=-1).cumsum(dim=-1)
            # keep every token up to and including the one that crosses p
            drop = cum - torch.softmax(srt, dim=-1) >= _p
            drop[..., 0] = False                      # always keep the argmax
            mask = torch.zeros_like(drop).scatter(-1, idx, drop)
            return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        _nucleus = _nucleus_proc
        print("[nucleus] top_p=%.3f active" % args.top_p, flush=True)

    # ---- constraint processor -------------------------------------------
    _proc = None
    if args.constrain != "none":
        import sys as _sys
        if D17_PATH not in _sys.path:
            _sys.path.insert(0, D17_PATH)
        from rxn_constraints import ReactionConstraints, build_comp_table
        _vocab = {t: i for i, t in inv_vocab.items()}
        _comp = None
        if args.constrain == "composition":
            _comp, _els = build_comp_table(_vocab, args.comp_cache, device)
            print("[constraints] elements: " + ",".join(_els), flush=True)
        _proc = ReactionConstraints(_vocab, device, mult_cap=args.mult_cap,
                                    composition=(args.constrain == "composition"),
                                    comp_table=_comp)
        print("[constraints] tier=%s mult_cap=%d" % (args.constrain, args.mult_cap),
              flush=True)

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="[topk]"):
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=(device.type == "cuda" and not args.fp32)):

                common = dict(
                    graph_x      = batch["graph_x"].to(device),
                    plddt        = batch["plddt"].to(device),
                    coords       = {k: v.to(device) for k, v in batch["coords"].items()},
                    torsions     = batch["torsions"].to(device),
                    edge_index   = batch["edge_index"].to(device),
                    batch_idx    = batch["batch_idx"].to(device),
                    protein_mask = batch["protein_mask"].to(device),
                    max_length   = args.max_length,
                    logits_processor = _compose(_proc, _nucleus),
                )
                _hyb = None
                if args.mode == "hybrid":
                    _nb = int(round(args.num_candidates * args.beam_frac))
                    _nb = max(1, min(args.num_candidates - 1, _nb))
                    _hyb = (_nb, args.num_candidates - _nb)
                    common.update(beam_width=_nb,
                                  length_normalization=args.length_norm,
                                  use_sampling=False)
                elif args.mode == "beam":
                    common.update(beam_width=args.num_candidates,
                                  length_normalization=args.length_norm,
                                  use_sampling=False)
                else:
                    common.update(beam_width=1, use_sampling=True,
                                  top_k=args.top_k, temperature=args.temperature,
                                  num_samples=args.num_candidates)

                if args.model_type == "rag":
                    common.update(
                        retrieved_token_ids = batch["retrieved_token_ids"].to(device),
                        retrieved_mask      = batch["retrieved_mask"].to(device),
                    )

                if _hyb is not None:
                    _nb, _ns = _hyb
                    preds, id_logits = model.predict(**common)
                    _cs = dict(common)
                    _cs.pop("length_normalization", None)
                    _cs.update(beam_width=1, use_sampling=True,
                               top_k=args.top_k, temperature=args.temperature,
                               num_samples=_ns)
                    _ps, _ = model.predict(**_cs)
                    _T = max(preds.size(-1), _ps.size(-1))

                    def _padc(x, T):
                        if x.size(-1) < T:
                            x = torch.cat(
                                [x, torch.full((x.size(0), x.size(1), T - x.size(-1)),
                                               PAD_ID, dtype=x.dtype, device=x.device)],
                                dim=-1)
                        return x

                    preds = torch.cat([_padc(preds, _T), _padc(_ps, _T)], dim=1)
                else:
                    preds, id_logits = model.predict(**common)

            assert preds.dim() == 3, f"expected 3D preds, got shape {preds.shape}"
            B, N, T = preds.shape

            for i in range(B):
                true_smi = "".join(
                    inv_vocab.get(t.item(), "")
                    for t in batch["labels"][i] if t.item() not in ignore
                )
                cand_smiles = [_decode_one(preds[i, n], inv_vocab,
                                           START_ID, END_ID, PAD_ID)
                               for n in range(N)]
                if args.id_override:
                    id_logits_row = id_logits[i:i+1].repeat(N, 1)
                    cand_smiles = decode_with_identity_override(
                        cand_smiles, id_logits_row, threshold=args.id_override_threshold,
                    )
                for rank in range(N):
                    pred = cand_smiles[rank]
                    t, valid = tanimoto(true_smi, pred)
                    rows.append({
                        "Model":          f"{args.model_type}+top{args.num_candidates}{args.mode}",
                        "row_idx":        len(rows) // N,
                        "rank":           rank,
                        "Tanimoto Score": round(t, 6),
                        "Valid SMILES":   valid,
                        "True SMILES":    true_smi,
                        "Pred SMILES":    pred,
                    })

    # fix row_idx — one unique id per input row
    n_inputs = len(rows) // args.num_candidates
    for r in range(n_inputs):
        for k in range(args.num_candidates):
            rows[r * args.num_candidates + k]["row_idx"] = r

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[topk-all] wrote {args.out_csv}  rows={len(df):,}  ({n_inputs} inputs x {args.num_candidates} candidates)")
    print(f"[topk-all] elapsed: {time.time()-t0:.1f}s")

    # --- summary ---
    print("\n========== TOP-K SUMMARY ==========")
    print(f"  inputs: {n_inputs}   candidates per input: {args.num_candidates}")
    top1 = df[df["rank"] == 0]
    topN = df.groupby("row_idx")["Tanimoto Score"].max().reset_index()

    for label, scores in [("top-1", top1["Tanimoto Score"]),
                          (f"top-{args.num_candidates}", topN["Tanimoto Score"])]:
        n = len(scores)
        print(f"  {label}: mean_T={scores.mean():.4f}  median_T={scores.median():.4f}  "
              f"T=1.00={(scores >= 0.999).sum():>5}  ({100*(scores >= 0.999).mean():.1f}%)  "
              f"T>=0.9={(scores >= 0.9).sum():>5}  ({100*(scores >= 0.9).mean():.1f}%)")

    t1_top1 = int((top1["Tanimoto Score"] >= 0.999).sum())
    t1_topN = int((topN["Tanimoto Score"] >= 0.999).sum())
    print(f"\n  top-{args.num_candidates} T=1.00 lift vs top-1: +{t1_topN - t1_top1} hits  "
          f"({100*(t1_topN - t1_top1)/max(n_inputs,1):.1f}pp)")


if __name__ == "__main__":
    main()
