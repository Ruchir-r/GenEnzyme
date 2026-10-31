#!/usr/bin/env python3
"""Optuna worker for cb_sweep_v2 — search for the BEST MODEL by product recall.

Difference from cb_sweep_v1 (which minimised validation loss):

  * OBJECTIVE = strict product recall on the em_ext VALIDATION set, measured by
    actually running free-running inference on the trained checkpoint and
    scoring exact isomeric-SMILES matches. v1's objective (val_main) is known
    to mis-rank: the ChemBERTa arm had the better val_main (2.2179 vs 2.5368)
    but significantly worse product recall (53 vs 81 strict hits @k=10,
    McNemar p=0.006). Loss is therefore used ONLY as a cheap pruning signal,
    never for final selection.

  * The test set is never touched. Search selects on val; test is reserved for
    the final reported number.

  * Search space covers architecture AND optimiser AND the embedding init.
    Decode settings are deliberately NOT Optuna dimensions: they need no
    retraining, so tuning them per-trial would waste GPU hours re-training
    identical models. Every trial is scored at one fixed reference decode
    config; the decode grid is swept afterwards over the top checkpoints
    (see decode_grid.py).

Usage:
  cb_sweep_v2_worker.py --study cb_sweep_v2 --sweep-dir <dir> --n-trials 8
"""
import argparse, glob, json, os, re, shutil, signal, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from recall_score import score_csv

R2 = "/path/to/struct_rxnpred"
D17 = os.path.join(R2, "claude_dir", "17_reaction_revival")
P15 = os.path.join(R2, "claude_dir", "15_p07_fair")
IDC = os.path.join(R2, "claude_dir", "07_id_curriculum")
EMD = os.path.join(R2, "data", "em_ext")
EMBEDS = "/path/to/scratch/embeddings"
PYTHON = os.path.expanduser("~/.conda/envs/torch-gpu-copy/bin/python3")

TRAIN_CSV = os.path.join(EMD, "em_ext_train.clean.csv")
VAL_CSV = os.path.join(EMD, "em_ext_val.clean.csv")
VOCAB = os.path.join(EMD, "vocab_em_ext.json")

DIMS = [256, 384, 512]

EPOCH_RE = re.compile(
    r"EPOCHCURVE epoch=(\d+) train_main=([0-9.eE+-]+) "
    r"val_main=([0-9.eE+-]+) val_id_acc=([0-9.eE+-]+)")

OOM_MARKERS = ("CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED",
               "torch.cuda.OutOfMemoryError")

INIT_PATTERNS = {
    "chembert": ["chembert_init_emext_d%d.pt", "chembert_init_em_ext_d%d.pt"],
    "morgan":   ["chem_init_em_ext_d%d.pt", "chem_init_emext_d%d.pt"],
}


def init_table(d_model, init):
    """Path to the embedding-init table, or None for random init."""
    if init == "random":
        return None
    for pat in INIT_PATTERNS[init]:
        cand = os.path.join(D17, "results", pat % d_model)
        if os.path.isfile(cand):
            return cand
    return os.path.join(D17, "results", INIT_PATTERNS[init][0] % d_model)


def _kill(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def run_streaming(cmd, env, cwd, log_path, deadline, on_line=None):
    """Run a subprocess, tee stdout to log_path, call on_line per line.
    Returns (returncode, saw_oom). Raises TimeoutError past deadline."""
    saw_oom = False
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, preexec_fn=os.setsid)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if any(m in line for m in OOM_MARKERS):
                    saw_oom = True
                if on_line is not None:
                    on_line(line)
                if time.time() > deadline:
                    _kill(proc)
                    raise TimeoutError("exceeded trial budget")
        finally:
            if proc.poll() is None:
                _kill(proc)
        proc.wait()
    return proc.returncode, saw_oom


def objective(trial, args):
    # ---- architecture -------------------------------------------------
    d_model = trial.suggest_categorical("d_model", DIMS)
    nhead = trial.suggest_categorical("nhead", [4, 8])
    n_dec = trial.suggest_categorical("num_decoder_layers", [2, 4, 6])
    ffn_mult = trial.suggest_categorical("ffn_mult", [2, 3, 4])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.2])
    # ---- embedding initialisation (ChemBERTa excluded: shown harmful) ---
    init = trial.suggest_categorical("init", ["morgan", "random"])
    # ---- optimiser ------------------------------------------------------
    base_lr = trial.suggest_float("base_lr", 5e-5, 6e-4, log=True)
    gnn_lr = trial.suggest_float("gnn_lr", 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    warmup = trial.suggest_categorical("warmup_frac", [0.03, 0.06, 0.10, 0.20])
    grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    focal_gamma = trial.suggest_categorical("focal_gamma", [0.0, 1.0, 2.0])
    id_w = trial.suggest_categorical("id_loss_weight", [0.0, 0.05, 0.1])

    tbl = init_table(d_model, init)
    if tbl is not None and not os.path.isfile(tbl):
        raise optuna.TrialPruned("missing %s init table: %s" % (init, tbl))

    ffn = ffn_mult * d_model
    tag = "%s_t%03d" % (args.tag_prefix, trial.number)
    ckpt_root = os.path.join(args.sweep_dir, "ckpt")
    log_path = os.path.join(args.sweep_dir, "logs", tag + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    trial.set_user_attr("run_tag", tag)
    trial.set_user_attr("decoder_ffn_dim", ffn)
    trial.set_user_attr("search_epochs", args.epochs)
    trial.set_user_attr("init_table", tbl or "random")
    trial.set_user_attr("log_path", log_path)

    arch_env = {
        "RXN_D_MODEL": str(d_model),
        "RXN_NHEAD": str(nhead),
        "RXN_NUM_DECODER_LAYERS": str(n_dec),
        "RXN_ENCODER_FFN_DIM": str(ffn),
        "RXN_DECODER_FFN_DIM": str(ffn),
        "RXN_DROPOUT": str(dropout),
    }
    env = os.environ.copy()
    env.update(arch_env)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(args.seed),
        "STRUCT_DIR": os.path.join(R2, "pdb_structures"),
        "WANDB_MODE": "offline",
        "WANDB_RUN_GROUP": args.study,
        "WANDB_DIR": args.sweep_dir,
    })

    deadline = time.time() + args.trial_timeout

    # ---- train -----------------------------------------------------------
    cmd = [PYTHON, "train.py",
           "--train-csv", TRAIN_CSV,
           "--val-csv", VAL_CSV,
           "--vocab-path", VOCAB,
           ] + ([] if tbl is None else ["--init-embed", tbl]) + [
           "--embeds-dir", EMBEDS,
           "--batch-size", str(batch_size),
           "--num-epochs", str(args.epochs),
           "--num-workers", "8",
           "--base-lr", "%.6g" % base_lr,
           "--gnn-lr", "%.6g" % gnn_lr,
           "--weight-decay", "%.6g" % weight_decay,
           "--warmup-frac", "%.4g" % warmup,
           "--grad-clip", "%.4g" % grad_clip,
           "--focal-gamma", "%.4g" % focal_gamma,
           "--id-loss-weight", "%.4g" % id_w,
           "--seed", str(args.seed),
           "--save-dir", ckpt_root,
           "--run-tag", tag,
           "--wandb-mode", "offline"]

    print("[worker] trial %d TRAIN init=%s d=%d h=%d dec=%d ffn=%d drop=%.2f "
          "lr=%.4g gnn=%.4g wd=%.4g warm=%.2f clip=%.2g bs=%d fg=%.2g idw=%.2g -> %s"
          % (trial.number, init, d_model, nhead, n_dec, ffn, dropout, base_lr,
             gnn_lr, weight_decay, warmup, grad_clip, batch_size, focal_gamma,
             id_w, tag), flush=True)

    state = {"last_val": None, "epochs": 0}

    def on_line(line):
        m = EPOCH_RE.search(line)
        if not m:
            return
        ep = int(m.group(1))
        val = float(m.group(3))
        state["last_val"] = val
        state["epochs"] = ep
        # Loss is a PRUNING signal only; selection happens on recall below.
        # The study direction is "maximize" (recall), so the intermediate
        # value handed to the pruner must also be higher-is-better. val_main
        # is a loss, so report its NEGATION -- reporting the raw loss would
        # invert MedianPruner and kill the low-loss (good) trials.
        trial.report(-val, ep)
        print("[worker] trial %d epoch %d val_main=%.6f" % (trial.number, ep, val),
              flush=True)
        if trial.should_prune():
            raise optuna.TrialPruned("median-pruned at epoch %d" % ep)

    try:
        rc, oom = run_streaming(cmd, env, IDC, log_path, deadline, on_line)
    except TimeoutError:
        raise optuna.TrialPruned("train exceeded %ds" % args.trial_timeout)
    if oom:
        raise optuna.TrialPruned("CUDA OOM at d=%d bs=%d" % (d_model, batch_size))
    if rc != 0:
        raise optuna.TrialPruned("train exited rc=%d" % rc)

    trial.set_user_attr("val_main", state["last_val"])
    trial.set_user_attr("epochs_done", state["epochs"])

    ckpt = os.path.join(ckpt_root, tag, "best_id_curriculum_model.pt")
    if not os.path.isfile(ckpt):
        raise optuna.TrialPruned("no checkpoint at %s" % ckpt)

    # ---- inference on VAL at the fixed reference decode config -----------
    infer_csv = os.path.join(args.sweep_dir, "infer", tag + "_val_%s%d.csv"
                             % (args.ref_mode, args.ref_num))
    os.makedirs(os.path.dirname(infer_csv), exist_ok=True)
    icmd = [PYTHON, "-u", "infer_topk_all.py",
            "--model-type", "p07",
            "--ckpt", ckpt,
            "--test-csv", VAL_CSV,
            "--vocab-path", VOCAB,
            "--embeds-dir", EMBEDS,
            "--out-csv", infer_csv,
            "--mode", args.ref_mode,
            "--num-candidates", str(args.ref_num),
            "--num-workers", "8",
            "--batch-size", "8",
            "--no-id-override"]
    print("[worker] trial %d INFER %s%d on val" % (trial.number, args.ref_mode,
                                                   args.ref_num), flush=True)
    try:
        rc, oom = run_streaming(icmd, env, P15, log_path, deadline)
    except TimeoutError:
        raise optuna.TrialPruned("inference exceeded trial budget")
    if rc != 0 or not os.path.isfile(infer_csv):
        raise optuna.TrialPruned("inference failed rc=%d" % rc)

    sc = score_csv(infer_csv)
    for k, v in sc.items():
        trial.set_user_attr("val_" + k, v)
    print("[worker] trial %d RECALL strict_topall=%.4f (%d/%d) strict_top1=%.4f "
          "const_topall=%.4f  val_main=%s"
          % (trial.number, sc["strict_topall"], sc["hits_strict_topall"], sc["n"],
             sc["strict_top1"], sc["const_topall"], state["last_val"]), flush=True)
    print("RESULT trial=%d init=%s d_model=%d nhead=%d dec=%d ffn=%d drop=%.2f "
          "lr=%.4g gnn_lr=%.4g wd=%.4g warm=%.2f clip=%.2g bs=%d fg=%.2g idw=%.2g "
          "val_main=%s val_strict_topall=%.4f val_strict_hits=%d tag=%s"
          % (trial.number, init, d_model, nhead, n_dec, ffn, dropout, base_lr,
             gnn_lr, weight_decay, warmup, grad_clip, batch_size, focal_gamma,
             id_w, state["last_val"], sc["strict_topall"],
             sc["hits_strict_topall"], tag), flush=True)

    # maximise strict product recall over the candidate set
    return sc["strict_topall"]


def prune_checkpoints(study, trial, sweep_dir, keep):
    """Keep only the top-`keep` trials' checkpoint dirs (objective = maximise)."""
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    best_tags = {t.user_attrs.get("run_tag") for t in
                 sorted(done, key=lambda t: -t.value)[:keep]}
    for path in sorted(glob.glob(os.path.join(sweep_dir, "ckpt", "*"))):
        if os.path.isdir(path) and os.path.basename(path) not in best_tags:
            shutil.rmtree(path, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="cb_sweep_v2")
    ap.add_argument("--sweep-dir", default=os.path.join(D17, "sweeps", "cb_sweep_v2"))
    ap.add_argument("--tag-prefix", default="cbv2")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=22 * 3600)
    ap.add_argument("--trial-timeout", type=int, default=5 * 3600)
    ap.add_argument("--keep-ckpt", type=int, default=4)
    ap.add_argument("--startup-trials", type=int, default=8)
    ap.add_argument("--ref-mode", default="sample", choices=["beam", "sample"],
                    help="decode mode used to score EVERY trial (fixed)")
    ap.add_argument("--ref-num", type=int, default=25,
                    help="candidate budget used to score every trial")
    args = ap.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)
    os.makedirs(os.path.join(args.sweep_dir, "logs"), exist_ok=True)

    storage = JournalStorage(
        JournalFileBackend(os.path.join(args.sweep_dir, args.study + ".journal")))
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=args.startup_trials),
        # conservative: loss-pruning can only kill clearly-bad runs, because
        # loss is a biased proxy for the recall objective
        # NOTE: intermediate values are reported as -val_main (see objective),
        # so that higher-is-better matches direction="maximize".
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=5),
    )
    study.set_user_attr("dataset", "em_ext")
    study.set_user_attr("objective",
                        "maximise strict product recall over %s%d candidates, em_ext VAL"
                        % (args.ref_mode, args.ref_num))
    study.set_user_attr("selection_set", "validation (test never touched)")

    print("[worker] host=%s study=%s existing_trials=%d ref=%s%d epochs=%d"
          % (os.uname().nodename, args.study, len(study.trials),
             args.ref_mode, args.ref_num, args.epochs), flush=True)

    def cb(st, tr):
        prune_checkpoints(st, tr, args.sweep_dir, args.keep_ckpt)

    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials,
                   timeout=args.timeout, callbacks=[cb],
                   catch=(RuntimeError,))
    print("[worker] done. trials=%d" % len(study.trials), flush=True)


if __name__ == "__main__":
    main()
