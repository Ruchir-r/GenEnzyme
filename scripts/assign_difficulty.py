#!/usr/bin/env python3
"""Assign each test reaction one difficulty class, relative to the training split.

Classes (same rule as scripts/bucket_score_cbv2.py, which reuses the classifier behind the published
1,275 / 1,074 split):
  exact_seen        reaction key (sorted reactant multiset, sorted product multiset) occurs in train
  recombination     key unseen, but every reactant and product metabolite occurs somewhere in train
  novel_metabolite  at least one metabolite never occurs in train
  unparseable       a component contains 'UNK' or fails RDKit parsing

Keys are MULTISETS (tuple(sorted(...))), never frozensets: stoichiometry is part of a reaction's identity.

usage: python scripts/assign_difficulty.py --train data/enzymemap_ext/train.csv \
           --test data/enzymemap_ext/test.csv --out data/enzymemap_ext/difficulty_classes.csv
"""
import argparse, csv, sys
from collections import Counter
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10 ** 8)


def parse_rxn(s):
    s = (s or "").strip()
    if ">>" not in s:
        return None
    left, right = (s.rsplit(">>", 1) if s.count(">>") > 1 else s.split(">>"))
    if ">" in left:
        left = left.split(">")[0]
    out = []
    for side in (left, right):
        comps = []
        for tok in side.split("."):
            tok = tok.strip()
            if not tok or "UNK" in tok:
                return None
            m = Chem.MolFromSmiles(tok)
            if m is None:
                return None
            comps.append(Chem.MolToSmiles(m, canonical=True))
        out.append(tuple(sorted(comps)))
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--col", default="rxnsmiles_ordered")
    a = ap.parse_args()
    keys, mets = set(), set()
    for r in csv.DictReader(open(a.train, newline="")):
        k = parse_rxn(r[a.col])
        if k is None:
            continue
        keys.add(k)
        mets.update(k[0]); mets.update(k[1])
    rows = list(csv.DictReader(open(a.test, newline="")))
    counts = Counter()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "Entry", "difficulty_class"])
        for i, r in enumerate(rows):
            k = parse_rxn(r[a.col])
            if k is None:
                b = "unparseable"
            elif k in keys:
                b = "exact_seen"
            elif set(k[0]) <= mets and set(k[1]) <= mets:
                b = "recombination"
            else:
                b = "novel_metabolite"
            counts[b] += 1
            w.writerow([i, r.get("Entry", ""), b])
    print("DIFFICULTY_COUNTS", dict(counts), "n =", len(rows))


if __name__ == "__main__":
    sys.exit(main())
