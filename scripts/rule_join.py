#!/usr/bin/env python3
"""Join em_ext reactions to EnzymeMap reaction RULES (templates) via a protonation/stereo-insensitive key.

key(reaction) = (sorted normalised reactant tokens, sorted normalised product tokens)
normalised token = RDKit Uncharger-neutralised, stereo-removed canonical SMILES; '[H+]' tokens dropped.
Matching is tried forward, then reversed (EnzymeMap lists each reaction once; em_ext may hold both directions).
Writes OUTPREFIX_rules.csv (split,row_idx,match,rule,ec,n_rules) and prints match rates.
"""
import csv, gzip, sys, collections, json
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10 ** 8)
R2 = '/path/to/struct_rxnpred'
EM = '/path/to/scratch/enzymemap_repo/data/processed_reactions.csv.gz'
unch = rdMolStandardize.Uncharger()
_M = object()
_c = {}


def ntok(tk):
    r = _c.get(tk, _M)
    if r is not _M:
        return r
    out = None
    if tk and len(tk) < 2000:
        m = Chem.MolFromSmiles(tk)
        if m is not None:
            try:
                m = unch.uncharge(m)
            except Exception:
                pass
            out = Chem.MolToSmiles(m, isomericSmiles=False)
    _c[tk] = out
    return out


def nside(s):
    out = []
    for t in s.split('.'):
        t = t.strip()
        if not t or t == '[H+]':
            continue
        n = ntok(t)
        if n is None:
            return None
        out.append(n)
    return tuple(sorted(out))


def key(rx):
    p = rx.split('>>')
    if len(p) != 2:
        return None
    a, b = nside(p[0]), nside(p[1])
    if a is None or b is None:
        return None
    return (a, b)


def main():
    prefix = sys.argv[1]
    fwd = collections.defaultdict(lambda: [set(), set()])
    n = 0
    with gzip.open(EM, 'rt') as f:
        for row in csv.DictReader(f):
            n += 1
            k = key(row['unmapped'])
            if k is None:
                continue
            fwd[k][0].add(row['rule_id']); fwd[k][1].add(row['ec_num'])
    print('enzymemap rows', n, 'keys', len(fwd), flush=True)
    out = []
    stat = collections.defaultdict(collections.Counter)
    for split in ('train', 'val', 'test'):
        rows = list(csv.DictReader(open(R2 + '/data/em_ext/em_ext_' + split + '.clean.csv')))
        for i, r in enumerate(rows):
            k = key(r['rxnsmiles_ordered'])
            match, rules, ecs = 'none', set(), set()
            if k is not None:
                if k in fwd:
                    match = 'fwd'; rules, ecs = fwd[k]
                elif (k[1], k[0]) in fwd:
                    match = 'rev'; rules, ecs = fwd[(k[1], k[0])]
            stat[split][match] += 1
            rule = (min(rules, key=lambda x: int(x)) + ('r' if match == 'rev' else '')) if rules else ''
            out.append([split, i, match, rule, ';'.join(sorted(ecs)), len(rules)])
    with open(prefix + '_rules.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['split', 'row_idx', 'match', 'rule', 'ec', 'n_rules']); w.writerows(out)
    print('MATCH', json.dumps({k: dict(v) for k, v in stat.items()}), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
