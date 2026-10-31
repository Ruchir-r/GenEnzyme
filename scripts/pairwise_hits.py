"""Dump per-row strict/const hit booleans for matched decode configs across
the seq / chem(Morgan) / chembert arms on em_ext, so arm differences can be
tested paired (McNemar) rather than as independent proportions.

Same canon_set/canon_flat logic as score_sweep3.py.
Writes a CSV: row_idx, <arm>_strict_top1, <arm>_strict_topall, ... per arm.
Usage: pairwise_hits.py <out.csv> <tag>=<csv> [<tag>=<csv> ...]
"""
import csv, sys, os, json
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10**8)

_cache_set, _cache_flat = {}, {}
MAX_TOK_LEN = 2000

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

def hits(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r['row_idx']), []).append(r)
    res = {}
    for idx, rs in rows.items():
        raw_t = rs[0].get('True SMILES') or rs[0].get('true_prod') or ''
        truth = canon_set(raw_t)
        if truth is None:
            continue
        truth_f = canon_flat(raw_t)
        preds = [canon_set(r.get('Pred SMILES') or r.get('pred_prod') or '') for r in rs]
        predf = [canon_flat(r.get('Pred SMILES') or r.get('pred_prod') or '') for r in rs]
        res[idx] = dict(
            s1=int(bool(preds) and preds[0] == truth),
            sA=int(truth in preds),
            c1=int(bool(predf) and predf[0] == truth_f),
            cA=int(truth_f in predf),
        )
    return res

if __name__ == '__main__':
    outp = sys.argv[1]
    specs = [a.split('=', 1) for a in sys.argv[2:]]
    per = {}
    for tag, path in specs:
        if not os.path.exists(path):
            print('MISSING ' + path, flush=True)
            continue
        per[tag] = hits(path)
        print('SCORED %s n=%d' % (tag, len(per[tag])), flush=True)
    tags = list(per)
    common = set.intersection(*[set(per[t]) for t in tags])
    print('COMMON_ROWS %d' % len(common), flush=True)
    fields = ['row_idx']
    for t in tags:
        fields += [t + '_s1', t + '_sA', t + '_c1', t + '_cA']
    with open(outp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for idx in sorted(common):
            row = [idx]
            for t in tags:
                h = per[t][idx]
                row += [h['s1'], h['sA'], h['c1'], h['cA']]
            w.writerow(row)
    print('WROTE %s rows=%d' % (outp, len(common)), flush=True)
