"""Bucket-stratified strict/const recall for cbv2_t010 sample pools.

Reuses hit_flags_wide.py's exact bucket classifier (parse_rxn, multiset
comparison against em_ext_train) so bucket assignment is IDENTICAL to the
already-published 1,275/1,074 recombination/novel_metabolite split -- no
re-derivation, just applying the same labels to a different model's pools.
"""
import csv, json, os, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10 ** 8)

R2 = "/path/to/struct_rxnpred"
EMD = R2 + "/data/em_ext"
D17 = R2 + "/claude_dir/17_reaction_revival"
FE = D17 + "/final_eval"

MAX_TOK_LEN = 2000
_cs = {}

def _canon_set_raw(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if '>>' in s:
        s = s.rsplit('>>', 1)[1]
    out = []
    for tok in s.split('.'):
        tok = tok.strip()
        if not tok or 'UNK' in tok or len(tok) > MAX_TOK_LEN:
            return None
        m = Chem.MolFromSmiles(tok)
        if m is None:
            return None
        out.append(Chem.MolToSmiles(m, isomericSmiles=True))
    return tuple(sorted(out)) if out else None

def canon_set(s):
    k = s if isinstance(s, str) else ''
    if k not in _cs:
        _cs[k] = _canon_set_raw(s)
    return _cs[k]

def parse_rxn(s):
    s = (s or '').strip()
    if '>>' not in s:
        return None
    left, right = (s.rsplit('>>', 1) if s.count('>>') > 1 else s.split('>>'))
    if '>' in left:
        left = left.split('>')[0]
    out = []
    for side in (left, right):
        comps = []
        for tok in side.split('.'):
            tok = tok.strip()
            if not tok or 'UNK' in tok:
                return None
            m = Chem.MolFromSmiles(tok)
            if m is None:
                return None
            comps.append(Chem.MolToSmiles(m, canonical=True))
        out.append(tuple(sorted(comps)))
    return out[0], out[1]

def load_col(path, col='rxnsmiles_ordered'):
    with open(path) as f:
        return [r[col] for r in csv.DictReader(f)]

tr_raw = load_col(EMD + '/em_ext_train.clean.csv')
te_raw = load_col(EMD + '/em_ext_test.clean.csv')

keys, mets = set(), set()
for s in tr_raw:
    k = parse_rxn(s)
    if k is None:
        continue
    keys.add(k)
    mets.update(k[0]); mets.update(k[1])

bucket_by_idx = {}
for i, s in enumerate(te_raw):
    k = parse_rxn(s)
    if k is None:
        bucket_by_idx[i] = 'unparseable'
    elif k in keys:
        bucket_by_idx[i] = 'exact_seen'
    elif set(k[0]) <= mets and set(k[1]) <= mets:
        bucket_by_idx[i] = 'recombination'
    else:
        bucket_by_idx[i] = 'novel_metabolite'

from collections import Counter
counts = Counter(bucket_by_idx.values())
print("BUCKET_COUNTS", dict(counts), flush=True)

POOLS = [
    ("sample25",  FE + "/cbv2_t010_test_sample25_T1_L1.csv"),
    ("sample50",  FE + "/cbv2_t010_test_sample50_T1_L1.csv"),
    ("sample100", FE + "/cbv2_t010_test_sample100_T1.2_L1.csv"),
    ("sample400_T1.2", FE + "/cbv2_t010_test_sample400_T1.2_L1.csv"),
    ("sample400_T1.5", FE + "/cbv2_t010_test_sample400_T1.5_L1.csv"),
]

rows_out = []
for tag, path in POOLS:
    if not os.path.exists(path):
        print("SKIP missing", tag, path, flush=True)
        continue
    per_row = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ri = int(r["row_idx"])
            per_row.setdefault(ri, []).append(r)

    bucket_hits = {b: {"n": 0, "top1": 0, "topall": 0} for b in
                   ("recombination", "novel_metabolite", "exact_seen", "unparseable")}
    for ri, rs in per_row.items():
        b = bucket_by_idx.get(ri, "unparseable")
        rs = sorted(rs, key=lambda r: int(r.get("rank", 0)))
        truth = canon_set(rs[0].get("True SMILES") or "")
        if truth is None:
            continue
        bucket_hits[b]["n"] += 1
        any_hit = False
        for j, r in enumerate(rs):
            cs = canon_set(r.get("Pred SMILES") or "")
            if cs is not None and cs == truth:
                any_hit = True
                if j == 0:
                    bucket_hits[b]["top1"] += 1
        if any_hit:
            bucket_hits[b]["topall"] += 1

    for b, d in bucket_hits.items():
        if d["n"] == 0:
            continue
        rec = dict(config=tag, bucket=b, n=d["n"],
                   top1_hits=d["top1"], top1_pct=100.0 * d["top1"] / d["n"],
                   topall_hits=d["topall"], topall_pct=100.0 * d["topall"] / d["n"])
        rows_out.append(rec)
        print("BUCKETRESULT %s" % json.dumps(rec), flush=True)

with open(D17 + "/results/cbv2_t010_bucketed.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
print("WROTE", D17 + "/results/cbv2_t010_bucketed.csv", flush=True)
