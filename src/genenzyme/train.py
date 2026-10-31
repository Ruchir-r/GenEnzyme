"""
Train GVPIdAwareMHGGAT with two modifications on top of plain GVP training:

(a) Identity-reaction auxiliary head (BCE on `is_identity` per sample).
(b) Focal-loss curriculum on the main token-level loss — up-weights hard
    examples in the gradient. This addresses the data-analysis finding
    that ~69% of samples are template-trivial (T=1.0) and dominate the
    gradient under plain CE; focal_gamma=2 down-weights well-classified
    tokens by `(1 - p_correct)^2` so the encoder/decoder spend more
    capacity on the hard 30%.

Total loss:
    L = focal_ce(token_logits, labels) + λ_id · bce(id_logit, is_identity)

Three optimizer groups:
    - base   : decoder, esm_proj, embeddings, modality, position
    - gnn    : featurizer, gat_layers, gnn_to_dmodel, gnn_gate, gnn_norms
    - id     : id_head (NEW; matches plot legend "id")

Warm-start
----------
--init-from points at best plain-GVP checkpoint. id_head starts random
(small init); other params load via load_state_dict(strict=False).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR  = os.path.abspath(os.path.join(_THIS_DIR, "..", "_lib"))
for p in (_LIB_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import DATA_CONFIG, TRAIN_CONFIG, SPECIAL_TOKENS
from data_curriculum import CurriculumGraphReactionDataset
from mhggat_id_aware import GVPIdAwareMHGGAT
from transformers import get_cosine_schedule_with_warmup

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


# =============================================================================
# Focal CE loss for the token-level objective
# =============================================================================
def focal_ce_loss(logits: torch.Tensor,
                  labels: torch.Tensor,
                  gamma: float = 2.0,
                  ignore_index: int = -100,
                  reduction: str = "mean") -> torch.Tensor:
    """
    Focal cross-entropy: loss_i = (1 - p_correct_i)^gamma * (-log p_correct_i)
    Down-weights tokens the model already predicts confidently. Tokens at
    `ignore_index` (e.g. PAD) are masked out.

    logits : [N, V]
    labels : [N]
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)         # [N, V]
    valid = (labels != ignore_index)                           # [N]

    # Gather per-token log-prob of the true class. Replace ignored labels
    # with 0 to avoid out-of-range gather indices; they get masked out below.
    safe_labels = labels.clamp(min=0)
    log_p_true = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [N]
    p_true = log_p_true.exp()
    focal_weight = (1.0 - p_true).pow(gamma)
    per_token_loss = -focal_weight * log_p_true                # [N]
    per_token_loss = per_token_loss * valid.float()

    if reduction == "sum":
        return per_token_loss.sum()
    if reduction == "none":
        return per_token_loss
    # mean over VALID tokens only (not over padded positions).
    n_valid = valid.float().sum().clamp(min=1.0)
    return per_token_loss.sum() / n_valid


# =============================================================================
# Eval — separate the two losses so we can track them
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, focal_gamma: float, id_loss_weight: float,
             device: str = "cuda") -> dict:
    model.eval()
    total_main, total_id, total_id_acc = 0.0, 0.0, 0.0
    n_batches = 0
    for batch in loader:
        if batch["edge_index"].numel() == 0:
            continue
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                            dtype=torch.bfloat16):
            main_logits, id_logit = model(
                graph_x=batch["graph_x"].to(device),
                plddt=batch["plddt"].to(device),
                coords={k: v.to(device) for k, v in batch["coords"].items()},
                torsions=batch["torsions"].to(device),
                edge_index=batch["edge_index"].to(device),
                batch_idx=batch["batch_idx"].to(device),
                protein_mask=batch["protein_mask"].to(device),
                reaction_ids=batch["reaction_ids"].to(device),
                reaction_mask=batch["reaction_mask"].to(device),
            )
            labels = batch["labels"].to(device)
            id_target = batch["is_identity"].to(device)

            main_loss = focal_ce_loss(
                main_logits.view(-1, main_logits.size(-1)),
                labels.view(-1),
                gamma=focal_gamma,
                ignore_index=SPECIAL_TOKENS["<PAD>"],
            )
            id_loss = F.binary_cross_entropy_with_logits(id_logit.float(), id_target)
            id_pred = (torch.sigmoid(id_logit) > 0.5).float()
            id_acc  = (id_pred == id_target).float().mean()

        total_main += main_loss.item()
        total_id += id_loss.item()
        total_id_acc += id_acc.item()
        n_batches += 1

    return {
        "main": total_main / max(n_batches, 1),
        "id":   total_id   / max(n_batches, 1),
        "id_acc": total_id_acc / max(n_batches, 1),
        "total": (total_main + id_loss_weight * total_id) / max(n_batches, 1),
    }


# =============================================================================
# Optimizer
# =============================================================================
def build_param_groups(model: nn.Module, base_lr: float, gnn_lr: float,
                       id_lr: float) -> list[dict]:
    id_params, gnn_params, base_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("id_head"):
            id_params.append(p)
        elif any(kw in name for kw in
                 ("featurizer", "gat_layers", "gnn_norms",
                  "gnn_gate", "gnn_to_dmodel", "gnn_final_norm")):
            gnn_params.append(p)
        else:
            base_params.append(p)
    return [
        {"params": base_params, "lr": base_lr, "name": "base"},
        {"params": gnn_params,  "lr": gnn_lr,  "name": "gnn"},
        {"params": id_params,   "lr": id_lr,   "name": "id"},
    ]


def warmstart_from(model: nn.Module, ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {"ckpt": ckpt_path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected)}


# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GVPIdAwareMHGGAT with focal-loss curriculum + id head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train-csv",  default=DATA_CONFIG["train_path"])
    p.add_argument("--val-csv",    default=DATA_CONFIG["val_path"])
    p.add_argument("--vocab-path", default=DATA_CONFIG["vocab_path"])
    p.add_argument("--embeds-dir", default=DATA_CONFIG["embeddings_dir"])

    # Model
    p.add_argument("--num-mpnn-layers", type=int, default=2)
    p.add_argument("--d-v",     type=int, default=16)
    p.add_argument("--num-rbf", type=int, default=16)
    p.add_argument("--init-from", default=None,
                   help="Warm-start checkpoint. Recommended: best_gvp_model.pt.")

    # Training
    p.add_argument("--init-embed", default=None,
                   help="path to a chem-init .pt from build_chem_init.py; "
                        "warm-starts shared_embedding.weight with ECFP4 + "
                        "stoichiometry/mass descriptor PCA. Stays trainable.")
    p.add_argument("--batch-size",  type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--num-epochs",  type=int, default=TRAIN_CONFIG["num_epochs"])
    p.add_argument("--num-workers", type=int, default=4)

    # Optimizer
    p.add_argument("--base-lr",  type=float, default=1e-4)
    p.add_argument("--gnn-lr",   type=float, default=5e-3)
    p.add_argument("--id-lr",    type=float, default=5e-3,
                   help="LR for id_head params. Faster so the auxiliary head "
                        "moves quickly when warm-starting from plain GVP.")
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--warmup-frac",      type=float, default=0.10)
    p.add_argument("--grad-clip",        type=float, default=1.0)

    # Loss
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal loss gamma. 0 = standard CE; 2 = strong "
                        "down-weighting of well-classified tokens.")
    p.add_argument("--id-loss-weight", type=float, default=0.1,
                   help="Coefficient on the BCE identity head loss in the "
                        "total objective.")

    # IO
    p.add_argument("--save-dir", default=os.path.join(_THIS_DIR, "checkpoints"))
    p.add_argument("--run-tag",  default="id_curriculum_warmstart")
    p.add_argument("--save-every", type=int, default=0,
                   help="additionally save epoch{N}_id_curriculum_model.pt every "
                        "N epochs (0=off). Needed to measure a delayed-"
                        "generalization (grokking) curve; best/last alone cannot.")
    p.add_argument("--wandb-mode", default="offline",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="If set, calls torch.manual_seed(seed) + numpy + python "
                        "random for reproducible 3-seed runs. PYTHONHASHSEED is "
                        "NOT enough on its own (it doesn't touch torch RNG).")
    return p.parse_args()


def _set_all_seeds(seed: int) -> None:
    """One-stop seeding for paper-grade variance bars."""
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA non-determinism still wins on some atomic ops; this gets us
    # close-to-reproducible, not bit-exact.


def main():
    args = parse_args()
    if args.seed is not None:
        _set_all_seeds(args.seed)
        print(f"[train-id-curriculum] seeded with torch/numpy/random seed={args.seed}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(args.save_dir, args.run_tag)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[train-id-curriculum] device={device}  save_dir={save_dir}")
    print(f"[train-id-curriculum] focal_gamma={args.focal_gamma}  "
          f"id_loss_weight={args.id_loss_weight}")

    # --- Datasets / loaders -------------------------------------------------
    train_ds = CurriculumGraphReactionDataset(args.train_csv, args.vocab_path, args.embeds_dir)
    val_ds   = CurriculumGraphReactionDataset(args.val_csv,   args.vocab_path, args.embeds_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=val_ds.collate_fn, num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    # Ensure the model's embedding/lm_head match the ACTUAL vocab used, not
    # config.py's hardcoded default (8882). mhggat.py reads
    # TRAIN_CONFIG["vocab_size"] inside GraphReactionPredictor.__init__ (at
    # construction time, not import time), so mutating it here — before the
    # model is built — is sufficient. Without this, a --vocab-path pointing
    # at a differently-sized vocab (e.g. the 2099/2370-token recomb vocabs)
    # would silently train against a mismatched/oversized output layer.
    _actual_vocab_size = len(train_ds.vocab)
    if TRAIN_CONFIG["vocab_size"] != _actual_vocab_size:
        print(f"[train-id-curriculum] overriding TRAIN_CONFIG['vocab_size'] "
              f"{TRAIN_CONFIG['vocab_size']} -> {_actual_vocab_size} "
              f"(from {args.vocab_path})")
        TRAIN_CONFIG["vocab_size"] = _actual_vocab_size

    # --- Model --------------------------------------------------------------
    model = GVPIdAwareMHGGAT(num_mpnn_layers=args.num_mpnn_layers,
                             d_v=args.d_v, num_rbf=args.num_rbf).to(device)
    if args.init_from:
        report = warmstart_from(model, args.init_from)
        print(f"[train-id-curriculum] warm-start from {report['ckpt']}")
        print(f"[train-id-curriculum]   missing keys (expected: id_head): "
              f"{len(report['missing_keys'])}")
        print(f"[train-id-curriculum]   unexpected keys: "
              f"{len(report['unexpected_keys'])}")

    if args.init_embed:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from chem_init_load import load_chem_init
        load_chem_init(model, args.init_embed)

    pgs = build_param_groups(model, base_lr=args.base_lr,
                             gnn_lr=args.gnn_lr, id_lr=args.id_lr)
    print(f"[train-id-curriculum] params: " +
          ", ".join(f"{g['name']}={sum(p.numel() for p in g['params']):,}" for g in pgs))

    optim = torch.optim.AdamW(pgs, weight_decay=args.weight_decay)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_frac), total_steps,
    )

    if _WANDB_OK and args.wandb_mode != "disabled":
        wandb.init(project="rxn_pred", name=args.run_tag, mode=args.wandb_mode,
                   config=vars(args))

    best_val = float("inf")
    for epoch in range(args.num_epochs):
        print(f"\n[epoch {epoch+1}/{args.num_epochs}]")
        model.train()
        ep_main, ep_id, ep_id_acc, n_b = 0.0, 0.0, 0.0, 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"train e{epoch+1}")
        for step, batch in pbar:
            if batch["edge_index"].numel() == 0:
                continue
            optim.zero_grad()
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                dtype=torch.bfloat16):
                main_logits, id_logit = model(
                    graph_x=batch["graph_x"].to(device),
                    plddt=batch["plddt"].to(device),
                    coords={k: v.to(device) for k, v in batch["coords"].items()},
                    torsions=batch["torsions"].to(device),
                    edge_index=batch["edge_index"].to(device),
                    batch_idx=batch["batch_idx"].to(device),
                    protein_mask=batch["protein_mask"].to(device),
                    reaction_ids=batch["reaction_ids"].to(device),
                    reaction_mask=batch["reaction_mask"].to(device),
                )
                labels = batch["labels"].to(device)
                id_target = batch["is_identity"].to(device)

                main_loss = focal_ce_loss(
                    main_logits.view(-1, main_logits.size(-1)),
                    labels.view(-1),
                    gamma=args.focal_gamma,
                    ignore_index=SPECIAL_TOKENS["<PAD>"],
                )
                id_loss = F.binary_cross_entropy_with_logits(id_logit.float(), id_target)
                loss = main_loss + args.id_loss_weight * id_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       max_norm=args.grad_clip)
            optim.step(); scheduler.step()

            with torch.no_grad():
                id_acc = ((torch.sigmoid(id_logit) > 0.5).float() == id_target).float().mean()
            ep_main += main_loss.item(); ep_id += id_loss.item()
            ep_id_acc += id_acc.item(); n_b += 1

            pbar.set_postfix({
                "main":  f"{main_loss.item():.4f}",
                "id":    f"{id_loss.item():.4f}",
                "id_a":  f"{id_acc.item():.3f}",
                "‖∇‖":   f"{grad_norm.item():.2f}",
            })
            if _WANDB_OK and args.wandb_mode != "disabled":
                wandb.log({
                    "train/main_loss": main_loss.item(),
                    "train/id_loss":   id_loss.item(),
                    "train/id_acc":    id_acc.item(),
                    "train/grad_norm": grad_norm.item(),
                    "lr/base": optim.param_groups[0]["lr"],
                    "lr/gnn":  optim.param_groups[1]["lr"],
                    "lr/id":   optim.param_groups[2]["lr"],
                    "global_step": epoch * len(train_loader) + step,
                })

        avg_main, avg_id, avg_id_acc = (
            ep_main / max(n_b, 1), ep_id / max(n_b, 1), ep_id_acc / max(n_b, 1),
        )
        val_metrics = evaluate(model, val_loader, args.focal_gamma,
                               args.id_loss_weight, device=device)
        print(f"[epoch {epoch+1}] "
              f"train main={avg_main:.4f}  id={avg_id:.4f}  id_acc={avg_id_acc:.3f} | "
              f"val main={val_metrics['main']:.4f}  id={val_metrics['id']:.4f}  "
              f"id_acc={val_metrics['id_acc']:.3f}")

        if _WANDB_OK and args.wandb_mode != "disabled":
            wandb.log({"epoch": epoch,
                       "train/epoch_main": avg_main,
                       "train/epoch_id":   avg_id,
                       "train/epoch_id_acc": avg_id_acc,
                       "val/main": val_metrics["main"],
                       "val/id":   val_metrics["id"],
                       "val/id_acc": val_metrics["id_acc"]})

        # Use main val_loss for best-model selection (id is auxiliary).
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch, "val_main": val_metrics["main"],
            "val_id":   val_metrics["id"], "val_id_acc": val_metrics["id_acc"],
            "args": vars(args),
        }, os.path.join(save_dir, "last_id_curriculum_model.pt"))
        if args.save_every and ((epoch + 1) % args.save_every == 0):
            _ep_path = os.path.join(
                save_dir, "epoch%04d_id_curriculum_model.pt" % (epoch + 1))
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_main": val_metrics["main"],
                "val_id":   val_metrics["id"], "val_id_acc": val_metrics["id_acc"],
                "args": vars(args),
            }, _ep_path)
            print("  saved periodic -> %s" % _ep_path, flush=True)
        print("EPOCHCURVE epoch=%d train_main=%.6f val_main=%.6f val_id_acc=%.4f"
              % (epoch + 1, avg_main, val_metrics["main"], val_metrics["id_acc"]),
              flush=True)

        if val_metrics["main"] < best_val:
            best_val = val_metrics["main"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_main": val_metrics["main"],
                "val_id":   val_metrics["id"], "val_id_acc": val_metrics["id_acc"],
                "args": vars(args),
            }, os.path.join(save_dir, "best_id_curriculum_model.pt"))
            print(f"  saved best -> {os.path.join(save_dir, 'best_id_curriculum_model.pt')}")


if __name__ == "__main__":
    main()
