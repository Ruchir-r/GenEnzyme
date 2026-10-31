#!/usr/bin/env python3
"""Stage B: sweep DECODE settings over an already-trained checkpoint.

Decode settings need no retraining, so they are swept here rather than as
Optuna dimensions — tuning them inside the search would re-train identical
models once per decode config.

The architecture env (RXN_*) must match the checkpoint; pass it via --arch-json
(the sweep worker records it per trial) or set the RXN_* vars in the caller.

Writes/appends one row per decode config to --out-csv.
"""
import argparse, csv, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recall_score import score_csv

R2 = "/path/to/struct_rxnpred"
P15 = os.path.join(R2, "claude_dir", "15_p07_fair")
EMD = os.path.join(R2, "data", "em_ext")
EMBEDS = "/path/to/scratch/embeddings"
PYTHON = os.path.expanduser("~/.conda/envs/torch-gpu-copy/bin/python3")
VOCAB = os.path.join(EMD, "vocab_em_ext.json")

# (mode, num_candidates, temperature, length_norm)
DEFAULT_GRID = [
    ("beam", 5, 1.0, 1.0),
    ("beam", 10, 1.0, 1.0),
    ("beam", 25, 1.0, 1.0),
    ("beam", 25, 1.0, 0.7),
    ("beam", 25, 1.0, 1.3),
    ("sample", 10, 1.0, 1.0),
    ("sample", 25, 1.0, 1.0),
    ("sample", 25, 0.7, 1.0),
    ("sample", 25, 1.3, 1.0),
    ("sample", 50, 1.0, 1.0),
    ("sample", 50, 0.7, 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-csv", default=os.path.join(EMD, "em_ext_val.clean.csv"),
                    help="default = VAL. Point at test ONLY for the final number.")
    ap.add_argument("--split-name", default="val")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--arch-json", default=None,
                    help="JSON dict of RXN_* env overrides matching the ckpt")
    ap.add_argument("--grid-json", default=None,
                    help="JSON list of [mode, num, temperature, length_norm]")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.arch_json:
        arch = json.load(open(args.arch_json)) if os.path.isfile(args.arch_json) \
            else json.loads(args.arch_json)
        env.update({k: str(v) for k, v in arch.items()})
        print("[grid] arch env: %s" % json.dumps(arch), flush=True)
    env["PYTHONUNBUFFERED"] = "1"
    env["STRUCT_DIR"] = os.path.join(R2, "pdb_structures")

    grid = DEFAULT_GRID
    if args.grid_json:
        grid = [tuple(x) for x in (json.load(open(args.grid_json))
                                   if os.path.isfile(args.grid_json)
                                   else json.loads(args.grid_json))]

    os.makedirs(args.work_dir, exist_ok=True)
    new = not os.path.isfile(args.out_csv)
    fh = open(args.out_csv, "a", newline="")
    writer = None

    for mode, num, temp, lnorm in grid:
        name = "%s_%s_%s%d_T%.2g_L%.2g" % (args.tag, args.split_name, mode, num, temp, lnorm)
        out_csv = os.path.join(args.work_dir, name + ".csv")
        t0 = time.time()
        if not os.path.isfile(out_csv):
            cmd = [PYTHON, "-u", "infer_topk_all.py",
                   "--model-type", "p07",
                   "--ckpt", args.ckpt,
                   "--test-csv", args.eval_csv,
                   "--vocab-path", VOCAB,
                   "--embeds-dir", EMBEDS,
                   "--out-csv", out_csv,
                   "--mode", mode,
                   "--num-candidates", str(num),
                   "--temperature", str(temp),
                   "--length-norm", str(lnorm),
                   "--num-workers", "8",
                   "--batch-size", os.environ.get("DG_BS", "8"),
                   "--no-id-override"]
            rc = subprocess.call(cmd, cwd=P15, env=env)
            if rc != 0 or not os.path.isfile(out_csv):
                print("[grid] FAILED %s rc=%d" % (name, rc), flush=True)
                continue
        sc = score_csv(out_csv)
        row = dict(tag=args.tag, split=args.split_name, mode=mode, num=num,
                   temperature=temp, length_norm=lnorm,
                   minutes=round((time.time() - t0) / 60.0, 2), **sc)
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                writer.writeheader()
        writer.writerow(row)
        fh.flush()
        print("GRIDRESULT %s" % json.dumps(row), flush=True)
    fh.close()
    print("[grid] wrote %s" % args.out_csv, flush=True)


if __name__ == "__main__":
    main()
