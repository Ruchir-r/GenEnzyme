"""Shared strict/constitutional product-recall scoring for em_ext inference CSVs.

strict = exact isomeric-SMILES match of the full product multiset
const  = same after stereochemistry is removed (constitutional match)

Identical canon logic to score_sweep3.py / score_chembert.py, factored out so
the sweep worker and the decode grid score identically.
"""
import csv, os
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10 ** 8)

MAX_TOK_LEN = 2000
_cache_set, _cache_flat = {}, {}


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
    key = s if isinstance(s, str) else ''
    if key not in _cache_set:
        _cache_set[key] = _canon_set_raw(s)
    return _cache_set[key]


def canon_flat(s):
    key = s if isinstance(s, str) else ''
    if key in _cache_flat:
        return _cache_flat[key]
    t = canon_set(s)
    if t is None:
        _cache_flat[key] = None
        return None
    flat = []
    for c in t:
        m = Chem.MolFromSmiles(c)
        if m is None:
            _cache_flat[key] = None
            return None
        Chem.RemoveStereochemistry(m)
        flat.append(Chem.MolToSmiles(m, isomericSmiles=False))
    _cache_flat[key] = tuple(sorted(flat))
    return _cache_flat[key]


def score_csv(path):
    """Return dict with n, strict/const top1 + topall rates (%) and raw counts."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r['row_idx']), []).append(r)
    n = bad = s1 = sA = c1 = cA = 0
    depths = []
    for idx in sorted(rows):
        rs = rows[idx]
        raw_t = rs[0].get('True SMILES') or rs[0].get('true_prod') or ''
        truth = canon_set(raw_t)
        if truth is None:
            bad += 1
            continue
        truth_f = canon_flat(raw_t)
        n += 1
        depths.append(len(rs))
        preds = [canon_set(r.get('Pred SMILES') or r.get('pred_prod') or '') for r in rs]
        predf = [canon_flat(r.get('Pred SMILES') or r.get('pred_prod') or '') for r in rs]
        if preds and preds[0] == truth:
            s1 += 1
        if truth in preds:
            sA += 1
        if predf and predf[0] == truth_f:
            c1 += 1
        if truth_f in predf:
            cA += 1
    p = lambda x: round(100.0 * x / max(n, 1), 4)
    return dict(csv=os.path.basename(path), n=n, unparseable_true=bad,
                max_cand=max(depths) if depths else 0,
                strict_top1=p(s1), strict_topall=p(sA),
                const_top1=p(c1), const_topall=p(cA),
                hits_strict_top1=s1, hits_strict_topall=sA,
                hits_const_top1=c1, hits_const_topall=cA)


if __name__ == '__main__':
    import sys, json
    for p_ in sys.argv[1:]:
        print(json.dumps(score_csv(p_)), flush=True)
