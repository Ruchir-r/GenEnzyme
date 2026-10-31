"""
Reformat this project's em_ext corpus into Horizyn's expected training schema.

Design (user's "retrain-reversed" comparison): Horizyn's architecture is
retrained from scratch on OUR data, so both models see identical training
reactions and the leakage picture is symmetric by construction.

Reaction identity uses the project's canonical ordered-multiset key
(matches topk_bucket_v2.py:parse_rxn and build_expanded_pool.py). Frozensets
are prohibited -- they collide 'A + A >> B' with 'A >> B'.

Emits, under <out>/:
  train_rxns.csv   rs_id, reaction_id, reaction_smiles   (unique TRAIN reactions)
  test_rxns.csv    rs_id, reaction_id, reaction_smiles   (unique VAL reactions)
  train_pairs.csv  pr_id, reaction_id, protein_id
  test_pairs.csv   pr_id, reaction_id, protein_id        (VAL pairs = in-training monitor)
  rxn_id_map.csv   reaction_id, reaction_smiles, in_train, in_val, in_test

The em_ext TEST split is deliberately NOT used here -- it is held out for the
reversed screening evaluation, exactly as the pretrained-checkpoint arm was.
"""
import argparse
import csv
import json
import h5py
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.MolToSmiles(m, canonical=True)


def parse_rxn(smi):
    """Canonical ordered-multiset reaction key. Mirrors topk_bucket_v2.py:parse_rxn."""
    if not isinstance(smi, str) or ">>" not in smi:
        return None
    if smi.count(">>") > 1:
        left, right = smi.rsplit(">>", 1)
    else:
        left, right = smi.split(">>")
    if ">" in left:
        left = left.split(">")[0]
    out = []
    for side in (left, right):
        comps = []
        for tok in side.split("."):
            tok = tok.strip()
            if not tok or "UNK" in tok:
                return None
            c = canon(tok)
            if c is None:
                return None
            comps.append(c)
        out.append(tuple(sorted(comps)))
    return (out[0], out[1])


def key_to_smiles(key):
    return ".".join(key[0]) + ">>" + ".".join(key[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--em-ext-dir", default="/path/to/struct_rxnpred/data/em_ext")
    ap.add_argument("--protein-h5", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.protein_h5, "r") as f:
        prot_ids = set(
            i.decode() if isinstance(i, bytes) else str(i) for i in f["ids"][:]
        )

    splits = {}
    for s in ("train", "val", "test"):
        splits[s] = pd.read_csv(f"{args.em_ext_dir}/em_ext_{s}.clean.csv")

    # Parse every row once; cache key per raw SMILES string
    cache = {}

    def key_of(smi):
        if smi not in cache:
            cache[smi] = parse_rxn(smi)
        return cache[smi]

    keysets = {}
    for s, df in splits.items():
        ks = set()
        for smi in df["rxnsmiles_ordered"]:
            k = key_of(smi)
            if k is not None:
                ks.add(k)
        keysets[s] = ks

    all_keys = sorted(keysets["train"] | keysets["val"] | keysets["test"])
    rid = {k: f"EM_{i:06d}" for i, k in enumerate(all_keys)}

    with open(f"{args.out}/rxn_id_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reaction_id", "reaction_smiles", "in_train", "in_val", "in_test"])
        for k in all_keys:
            w.writerow([rid[k], key_to_smiles(k),
                        int(k in keysets["train"]), int(k in keysets["val"]),
                        int(k in keysets["test"])])

    def write_rxns(path, keys):
        keys = sorted(keys)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rs_id", "reaction_id", "reaction_smiles"])
            for i, k in enumerate(keys):
                w.writerow([i, rid[k], key_to_smiles(k)])
        return len(keys)

    n_train_rxn = write_rxns(f"{args.out}/train_rxns.csv", keysets["train"])
    n_val_rxn = write_rxns(f"{args.out}/test_rxns.csv", keysets["val"])

    stats = {}

    def write_pairs(path, df, split_name):
        seen = set()
        rows = []
        drop_rxn = drop_prot = dup = 0
        for entry, smi in zip(df["Entry"], df["rxnsmiles_ordered"]):
            k = key_of(smi)
            if k is None:
                drop_rxn += 1
                continue
            if entry not in prot_ids:
                drop_prot += 1
                continue
            pair = (rid[k], entry)
            if pair in seen:
                dup += 1
                continue
            seen.add(pair)
            rows.append(pair)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pr_id", "reaction_id", "protein_id"])
            for i, (r, p) in enumerate(rows):
                w.writerow([i, r, p])
        stats[split_name] = {
            "rows_in": int(len(df)), "pairs_out": len(rows),
            "dropped_unparseable_rxn": drop_rxn,
            "dropped_protein_not_embedded": drop_prot,
            "dropped_duplicate_pair": dup,
        }
        return len(rows)

    write_pairs(f"{args.out}/train_pairs.csv", splits["train"], "train")
    write_pairs(f"{args.out}/test_pairs.csv", splits["val"], "val")

    # Structural checks the comparison depends on
    ov_tr_te = len(keysets["train"] & keysets["test"])
    ov_tr_va = len(keysets["train"] & keysets["val"])

    print("EMEXT_HZ_DATA_JSON", json.dumps({
        "unique_keys_train": len(keysets["train"]),
        "unique_keys_val": len(keysets["val"]),
        "unique_keys_test": len(keysets["test"]),
        "pool_total_unique": len(all_keys),
        "train_rxns_written": n_train_rxn,
        "test_rxns_written": n_val_rxn,
        "pairs": stats,
        "rxnkey_overlap_train_test": ov_tr_te,
        "rxnkey_overlap_train_val": ov_tr_va,
        "proteins_embedded": len(prot_ids),
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
