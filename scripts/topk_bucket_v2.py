"""
Bucket-resolved top-k hit rates: ALL splits (incl. random) x ALL arms (incl. CLIPZyme)
x ALL decode variants (beam5, sample10, beam25, union).

Generalizes topk_bucket_all.py:
  - adds the RANDOM split (data/filtered2_{train,test}.csv) as the leaky reference point
  - adds the clipzyme arm (frozen CLIPZyme 1280-d protein embeddings)
  - adds beam25, including rank-truncated beam25@5 (ranks 0-4) which is the
    apples-to-apples comparison against beam5: same k, wider search
  - emits an "ALL" pseudo-bucket row so corpus-wide and bucket-resolved
    numbers live in one table and reconcile by construction

Buckets (each test reaction vs its matched TRAIN split), identical to recomb_slice.py:
  exact_seen        : (reactant-multiset, product-multiset) present in train
  recombination     : not exact-seen, but every metabolite seen in train
  novel_metabolite  : >=1 metabolite never seen in train

Hit criterion: product Tanimoto == 1.0.
NO GPU: scores already-computed candidate CSVs.
"""
import csv, os, glob
from collections import defaultdict
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit import Chem
csv.field_size_limit(10**7)

REPO = "/path/to/struct_rxnpred"
R  = f"{REPO}/claude_dir/05_inference_compare/results"
LC = f"{REPO}/claude_dir/09_benchmarks/results"
HIT = 0.9999

_CANON = {}

def canon(smi):
    if smi in _CANON: return _CANON[smi]
    m = Chem.MolFromSmiles(smi)
    out = Chem.MolToSmiles(m, canonical=True) if m is not None else None
    _CANON[smi] = out; return out

def parse_rxn(rxn):
    if not isinstance(rxn, str) or ">>" not in rxn: return None
    left, right = (rxn.split(">>",1) if rxn.count(">>")==1
                   else (rxn.rsplit(">>",1)[0], rxn.rsplit(">>",1)[1]))
    left = left.split(">")[0] if ">" in left and ">>" not in left else left
    reactants, products = [], []
    for side, bucket in ((left, reactants), (right, products)):
        for tok in side.split("."):
            tok = tok.strip()
            if not tok or "UNK" in tok: return None
            c = canon(tok)
            if c is None: return None
            bucket.append(c)
    return tuple(sorted(reactants)), tuple(sorted(products))

def load_train(path):
    keys, mets = set(), set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            p = parse_rxn(row.get("rxnsmiles_ordered",""))
            if p is None: continue
            r, pr = p; keys.add((r,pr)); mets.update(r); mets.update(pr)
    return keys, mets

def classify(true_rxn, keys, mets):
    p = parse_rxn(true_rxn)
    if p is None: return "unparseable"
    r, pr = p
    if (r,pr) in keys: return "exact_seen"
    if (set(r)|set(pr)) <= mets: return "recombination"
    return "novel_metabolite"

def load_candidates(path):
    """row_idx -> {'true': smiles, 'ranks': {rank: tanimoto}}"""
    d = defaultdict(lambda: {"true": None, "ranks": {}})
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ri = int(row["row_idx"]); rk = int(row["rank"])
            try: t = float(row["Tanimoto Score"])
            except (TypeError, ValueError): t = 0.0
            d[ri]["true"] = row.get("True SMILES")
            d[ri]["ranks"][rk] = t
    return d

def hit_at(ranks, kmax=None):
    """any hit among ranks < kmax (kmax=None -> all ranks)"""
    if kmax is None:
        return any(v >= HIT for v in ranks.values())
    return any(v >= HIT for k, v in ranks.items() if k < kmax)

# split -> (train_csv, test_csv)
SPLITS = {
    "random":     (f"{REPO}/data/filtered2_train.csv",              f"{REPO}/data/filtered2_test.csv"),
    "simsplit30": (f"{LC}/lencheck_p07_simsplit30_train.clean.csv", f"{LC}/lencheck_p07_simsplit30_test.clean.csv"),
    "simsplit40": (f"{LC}/lencheck_p07_simsplit40_train.clean.csv", f"{LC}/lencheck_p07_simsplit40_test.clean.csv"),
    "simsplit50": (f"{LC}/lencheck_p07_simsplit50_train.clean.csv", f"{LC}/lencheck_p07_simsplit50_test.clean.csv"),
}
# canonical arm -> filename aliases (random-split runs used short names)
ARMS = {
    "seq":         ["seq"],
    "p07fair":     ["p07fair", "p07"],
    "ragfairwarm": ["ragfairwarm", "rag"],
    "clipzyme":    ["clipzyme"],
}
CATS = ["exact_seen", "recombination", "novel_metabolite"]

def find_csv(split, arm, tag):
    for alias in ARMS[arm]:
        p = f"{R}/inference_detailed_{split}_{alias}_{tag}.csv"
        if os.path.exists(p): return p
    return None

METRICS = ["beam5_top1","beam5_top5","samp10_top1","samp10_top10",
           "union15_top1","union15_any","beam25_top1","beam25_at5","beam25_top25"]

rows_out = []
for split, (train_path, _test_path) in SPLITS.items():
    if not os.path.exists(train_path):
        print(f"[{split}] SKIP - no train csv", flush=True); continue
    print(f"[{split}] loading train ...", flush=True)
    keys, mets = load_train(train_path)
    print(f"[{split}] train: {len(keys)} rxns, {len(mets)} metabolites", flush=True)

    bucket = None
    for arm in ARMS:
        bp  = find_csv(split, arm, "beam5")
        sp  = find_csv(split, arm, "sample10")
        b25 = find_csv(split, arm, "beam25")
        if not any([bp, sp, b25]):
            continue
        beam   = load_candidates(bp)  if bp  else {}
        sample = load_candidates(sp)  if sp  else {}
        wide   = load_candidates(b25) if b25 else {}
        # row universe = union of whatever decodes exist
        ridx = sorted(set(beam) | set(sample) | set(wide))
        truth = {}
        for i in ridx:
            for src in (beam, sample, wide):
                if i in src and src[i]["true"] is not None:
                    truth[i] = src[i]["true"]; break

        if bucket is None:
            bucket = {i: classify(truth.get(i,""), keys, mets) for i in ridx}
            bc = defaultdict(int)
            for v in bucket.values(): bc[v] += 1
            print(f"[{split}] buckets: {dict(bc)}", flush=True)
        else:
            mism = sum(1 for i in ridx
                       if i in bucket and classify(truth.get(i,""), keys, mets) != bucket[i])
            assert mism == 0, f"{split}/{arm}: {mism} row_idx bucket mismatches"

        have = {"beam5": bool(bp), "samp10": bool(sp), "beam25": bool(b25)}
        have["union15"] = bool(bp and sp)

        agg = {c: defaultdict(int) for c in CATS + ["ALL"]}
        for i in ridx:
            c = bucket.get(i)
            if c not in CATS: continue
            for key in (c, "ALL"):
                a = agg[key]; a["n"] += 1
                br = beam.get(i,{}).get("ranks",{})
                sr = sample.get(i,{}).get("ranks",{})
                wr = wide.get(i,{}).get("ranks",{})
                a["beam5_top1"]   += int(br.get(0,0.0) >= HIT)
                a["beam5_top5"]   += int(hit_at(br))
                a["samp10_top1"]  += int(sr.get(0,0.0) >= HIT)
                a["samp10_top10"] += int(hit_at(sr))
                a["union15_top1"] += int(br.get(0,0.0) >= HIT or sr.get(0,0.0) >= HIT)
                a["union15_any"]  += int(hit_at(br) or hit_at(sr))
                a["beam25_top1"]  += int(wr.get(0,0.0) >= HIT)
                a["beam25_at5"]   += int(hit_at(wr, 5))
                a["beam25_top25"] += int(hit_at(wr))

        for c in CATS + ["ALL"]:
            a = agg[c]; n = a["n"]
            if not n: continue
            rec = {"split": split, "arm": arm, "bucket": c, "n": n}
            for m in METRICS:
                src = ("beam25" if m.startswith("beam25")
                       else "union15" if m.startswith("union15")
                       else "samp10" if m.startswith("samp10") else "beam5")
                if have.get(src):
                    rec[m] = round(a[m]/n, 4); rec[m+"_hits"] = a[m]
                else:
                    rec[m] = ""; rec[m+"_hits"] = ""
            rows_out.append(rec)
            print(f"  {arm:11} {c:16} n={n:6d} | b5_1={rec['beam5_top1']} b5_5={rec['beam5_top5']}"
                  f" | s_1={rec['samp10_top1']} s_10={rec['samp10_top10']}"
                  f" | U15={rec['union15_any']} | b25_1={rec['beam25_top1']}"
                  f" b25@5={rec['beam25_at5']} b25_25={rec['beam25_top25']}", flush=True)

out = f"{REPO}/rxn_recomb/results/topk_bucket_v2.csv"
fields = ["split","arm","bucket","n"] + [m for mm in METRICS for m in (mm, mm+"_hits")]
with open(out,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows_out)
print(f"\nwrote {out}  ({len(rows_out)} rows)")
