#!/usr/bin/env python3
"""RxnPred sequence-only baseline on em_ext (EnzymeMap), reproducing the
published architecture/optimiser settings.

Architecture (manuscript Methods): 4 encoder layers, 6 reaction-decoder
layers, 4 attention heads, d_model=320, FFN 4096 (encoder) / 2048 (decoder).
Protein input is the ESM2 t12-35M-UR50D per-residue representation (480-d,
layer 12, first 512 residues, FP16) -- the SEQUENCE-ONLY arm, i.e. no ESM-IF1
structure embedding and therefore 480-d rather than the concatenated 992-d.
Metabolite-level tokenisation with a closed vocabulary; token embedding
initialised from precomputed ChemBERTa vectors.

Optimiser: AdamW, peak lr 4.4e-4, cosine annealing, linear warmup over the
first 10% of steps, 25 epochs, effective batch size 128.

Uses this repo's ReactionPredictor (src/model.py) and ReactionDataset
(src/data.py) so the result is scored by the same pipeline as every other arm
in the comparison.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

REPO = "/path/to/struct_rxnpred"
# ORDER IS LOAD-BEARING: src/ supplies ReactionPredictor and ReactionDataset,
# but BOTH src/config.py and claude_dir/_lib/config.py define TRAIN_CONFIG and
# only the _lib one reads the RXN_* environment overrides. src/ is inserted
# first so that _lib, inserted after, ends up EARLIER on sys.path and wins the
# bare `from config import ...` inside src/model.py. Reversing these two lines
# silently trains at the d_model=256 defaults and then dies on the init-table
# dimension check.
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, REPO + "/claude_dir/_lib")

from config import TRAIN_CONFIG, MODEL_CONFIG, SPECIAL_TOKENS  # noqa: E402


def load_chem_init(model, path):
    """Copy a precomputed [vocab, d_model] table into shared_embedding.

    NOTE: shared_embedding is TIED to the output projection in this
    architecture, so it is loaded-and-trained rather than frozen. RxnPred
    freezes its metabolite embedding; that is not expressible here without
    also freezing the unembedding, which RxnPred does not do. Recorded as the
    single architectural deviation from the published setup.
    """
    tab = torch.load(path, map_location="cpu")
    if isinstance(tab, dict):
        tab = tab.get("embedding", tab.get("weight", tab))
    emb = model.shared_embedding.weight.data
    n = min(emb.shape[0], tab.shape[0])
    assert emb.shape[1] == tab.shape[1], (
        "init table dim %d != d_model %d" % (tab.shape[1], emb.shape[1]))
    emb[:n] = tab[:n].to(emb.dtype)
    print("[init] loaded %d/%d rows from %s" % (n, emb.shape[0], path), flush=True)


def evaluate(model, loader, crit, device):
    model.eval()
    tot = ntok = 0.0
    with torch.no_grad():
        for b in loader:
            logits = model(b["protein_embedding"].to(device).float(),
                           b["protein_mask"].to(device),
                           b["reaction_ids"].to(device),
                           b["reaction_mask"].to(device))
            lab = b["labels"].to(device)
            loss = crit(logits.reshape(-1, logits.size(-1)), lab.reshape(-1))
            m = (lab != SPECIAL_TOKENS["<PAD>"]).sum().item()
            tot += loss.item() * m
            ntok += m
    model.train()
    return tot / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--vocab-path", required=True)
    ap.add_argument("--embeds-dir", required=True)
    ap.add_argument("--init-embed", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="per-device 64 x 2 = effective 128, as published")
    ap.add_argument("--num-epochs", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=4.4e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.10)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--run-tag", default="rxnpred_seq")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from data import ReactionDataset
    tr = ReactionDataset(args.train_csv, args.vocab_path, args.embeds_dir)
    va = ReactionDataset(args.val_csv, args.vocab_path, args.embeds_dir)

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    TRAIN_CONFIG["vocab_size"] = max(vocab.values()) + 1

    from model import ReactionPredictor
    model = ReactionPredictor().to(device)
    if args.init_embed:
        load_chem_init(model, args.init_embed)

    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[cfg] d_model=%d nhead=%d enc=%d dec=%d ffn_enc=%d ffn_dec=%d "
          "input_dim=%d vocab=%d params=%.2fM"
          % (TRAIN_CONFIG["d_model"], TRAIN_CONFIG["nhead"],
             TRAIN_CONFIG["num_encoder_layers"],
             TRAIN_CONFIG["num_decoder_layers"],
             TRAIN_CONFIG["encoder_ffn_dim"], TRAIN_CONFIG["decoder_ffn_dim"],
             MODEL_CONFIG["input_dim"], TRAIN_CONFIG["vocab_size"],
             nparam / 1e6), flush=True)

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    collate_fn=tr.collate_fn, num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"), drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                    collate_fn=va.collate_fn, num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"))

    crit = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<PAD>"])
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(tl) // args.grad_accum)
    total = steps_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim, int(args.warmup_frac * total), total)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, args.run_tag)
    os.makedirs(ckpt, exist_ok=True)
    best = float("inf")
    t0 = time.time()

    for ep in range(1, args.num_epochs + 1):
        run_loss = run_tok = 0.0
        optim.zero_grad(set_to_none=True)
        for i, b in enumerate(tl):
            logits = model(b["protein_embedding"].to(device).float(),
                           b["protein_mask"].to(device),
                           b["reaction_ids"].to(device),
                           b["reaction_mask"].to(device))
            lab = b["labels"].to(device)
            loss = crit(logits.reshape(-1, logits.size(-1)), lab.reshape(-1))
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step(); scheduler.step()
                optim.zero_grad(set_to_none=True)
            m = (lab != SPECIAL_TOKENS["<PAD>"]).sum().item()
            run_loss += loss.item() * m; run_tok += m
        vloss = evaluate(model, vl, crit, device)
        print("EPOCHCURVE epoch=%d train_main=%.4f val_main=%.4f lr=%.3e elapsed=%.1f"
              % (ep, run_loss / max(run_tok, 1), vloss,
                 scheduler.get_last_lr()[0], time.time() - t0), flush=True)
        if vloss < best:
            best = vloss
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": ep, "val_main": vloss,
                        "train_config": dict(TRAIN_CONFIG),
                        "model_config": dict(MODEL_CONFIG)},
                       os.path.join(ckpt, "best_id_curriculum_model.pt"))

    print("RESULT run_tag=%s best_val=%.4f epochs=%d params=%.2fM"
          % (args.run_tag, best, args.num_epochs, nparam / 1e6), flush=True)


if __name__ == "__main__":
    main()
