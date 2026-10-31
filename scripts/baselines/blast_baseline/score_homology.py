#!/usr/bin/env python3
"""Score a BLASTp/DIAMOND tabular result as reaction-pool product recall@k,
the sequence-homology analogue of Horizyn's reversed screening.

Protocol (matches Horizyn's enzyme-query -> ranked-reaction-pool exactly,
substituting sequence bitscore for the learned dual-encoder score):

  For each test reaction row (test enzyme E_q, true product P):
    1. rank train enzymes by descending bitscore of their best HSP vs E_q
       (BLAST/DIAMOND outfmt6; keep max bitscore per (query,target) pair).
    2. walk train enzymes best-first; each contributes its train-pool products
       in a fixed order (most-frequent-first is irrelevant here -- we only ask
       whether P appears within the first k DISTINCT products encountered).
    3. recall@k = 1 if P is among the first k distinct products, else 0.

  strict = P matched as the exact canonical isomeric-SMILES multiset key
  const  = P matched after stereochemistry is flattened (same relaxation the
           generative-model table's "const" column uses)

Self-hits are KEPT: em_ext is reaction-disjoint but not enzyme-disjoint, so a
100%-identity hit to the same enzyme's OTHER training reactions is exactly what
homology transfer does. Filtering them would be a different (enzyme-disjoint)
experiment; reported as-is.

Usage: score_homology.py <blast_outfmt6> <train_pool.json> <test_truth.json> \
                         <out_prefix> <tool_label>
"""
import csv, json, sys
from collections import defaultdict

KS = [1, 5, 10, 15, 25, 50, 100, 200]


def flat(key):
    """stereo-flattened product key: strip @ / and isomeric marks crudely by
    re-canonicalising without stereo. Done via RDKit to match const semantics."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in key:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        out.append(Chem.MolToSmiles(m, isomericSmiles=False))
    return tuple(sorted(out))


def main():
    hits_path, pool_path, truth_path, out_prefix, label = sys.argv[1:6]
    pool = {e: [tuple(k) for k in v] for e, v in json.load(open(pool_path)).items()}
    truth = json.load(open(truth_path))

    # best bitscore per (query, target)
    best = defaultdict(dict)   # query -> {target: bitscore}
    with open(hits_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            q, t, bits = p[0], p[1], float(p[11])
            if bits > best[q].get(t, -1):
                best[q][t] = bits

    # precompute flattened pool products for const scoring
    flatpool = {}
    for e, keys in pool.items():
        fl = []
        for k in keys:
            fk = flat(k)
            if fk and fk not in fl:
                fl.append(fk)
        flatpool[e] = fl

    n = len(truth)
    n_hit_any = 0
    strict = {k: 0 for k in KS}
    const = {k: 0 for k in KS}
    rows = []
    for t in truth:
        E, P = t["entry"], t["truth"]
        if P is None:
            rows.append(dict(row=t["row"], entry=E, found_strict=0, found_const=0))
            continue
        P = tuple(P)
        Pf = flat(P)
        ranked = sorted(best.get(E, {}).items(), key=lambda kv: -kv[1])
        if ranked:
            n_hit_any += 1
        # walk enzymes best-first, accumulate distinct products (strict + const)
        seen_s, seen_f = [], []
        fs = fc = None
        for tgt, _ in ranked:
            for k in pool.get(tgt, []):
                if k not in seen_s:
                    seen_s.append(k)
                    if fs is None and k == P:
                        fs = len(seen_s)
            for fk in flatpool.get(tgt, []):
                if fk not in seen_f:
                    seen_f.append(fk)
                    if fc is None and Pf is not None and fk == Pf:
                        fc = len(seen_f)
            if fs is not None and fc is not None:
                break
        for k in KS:
            if fs is not None and fs <= k:
                strict[k] += 1
            if fc is not None and fc <= k:
                const[k] += 1
        rows.append(dict(row=t["row"], entry=E,
                         found_strict=fs or 0, found_const=fc or 0))

    print("TOOL", label, "n", n, "queries_with_any_hit", n_hit_any)
    print("k    strict%   strict_hits   const%   const_hits")
    curve = []
    for k in KS:
        sp = 100.0 * strict[k] / n
        cp = 100.0 * const[k] / n
        curve.append(dict(tool=label, k=k, n=n,
                          strict_pct=round(sp, 4), strict_hits=strict[k],
                          const_pct=round(cp, 4), const_hits=const[k]))
        print("%-4d %8.4f %12d %8.4f %11d" % (k, sp, strict[k], cp, const[k]))
    with open(out_prefix + "_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
    with open(out_prefix + "_perrow.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("WROTE", out_prefix + "_{curve,perrow}.csv")


if __name__ == "__main__":
    main()
