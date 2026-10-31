import sys, csv, json
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

RD = '/path/to/struct_rxnpred/claude_dir/05_inference_compare/results'
ARMS = {'fair': 'structure (GVP)', 'seq24': 'sequence only (24 ep)'}
DECS = ['beam5', 'sample10']
csv.field_size_limit(10**8)

def canon_set(smi, iso):
    if not smi or not isinstance(smi, str):
        return None
    s = smi.strip()
    if not s or 'UNK' in s:
        return None
    if '>>' in s:
        s = s.rsplit('>>', 1)[1]
    parts = []
    for tok in s.split('.'):
        tok = tok.strip()
        if not tok:
            continue
        m = Chem.MolFromSmiles(tok)
        if m is None:
            return None
        parts.append(Chem.MolToSmiles(m, canonical=True, isomericSmiles=iso))
    if not parts:
        return None
    return tuple(sorted(parts))

out = []
for arm in ARMS:
    for dec in DECS:
        path = RD + '/inference_detailed_emext_' + arm + '_' + dec + '.csv'
        rows = defaultdict(list)
        truth = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                ri = int(r['row_idx'])
                rows[ri].append(r['Pred SMILES'])
                truth[ri] = r['True SMILES']

        n = len(rows)
        k1_strict = k1_const = 0
        kA_strict = kA_const = 0
        unparse_true = 0
        for ri, preds in rows.items():
            t_iso = canon_set(truth[ri], True)
            t_con = canon_set(truth[ri], False)
            if t_iso is None:
                unparse_true += 1
                continue
            p_iso = [canon_set(p, True) for p in preds]
            p_con = [canon_set(p, False) for p in preds]
            if p_iso and p_iso[0] == t_iso:
                k1_strict += 1
            if p_con and p_con[0] == t_con:
                k1_const += 1
            if any(x == t_iso for x in p_iso):
                kA_strict += 1
            if any(x == t_con for x in p_con):
                kA_const += 1

        d = dict(arm=arm, arm_label=ARMS[arm], decoder=dec, n=n,
                 n_scored=n - unparse_true, unparseable_true=unparse_true,
                 k1_strict=k1_strict, k1_const=k1_const,
                 kany_strict=kA_strict, kany_const=kA_const)
        sc = max(d['n_scored'], 1)
        d['k1_strict_pct'] = round(100.0 * k1_strict / sc, 3)
        d['k1_const_pct'] = round(100.0 * k1_const / sc, 3)
        d['kany_strict_pct'] = round(100.0 * kA_strict / sc, 3)
        d['kany_const_pct'] = round(100.0 * kA_const / sc, 3)
        out.append(d)
        print('RESULT', json.dumps(d), flush=True)

with open(RD + '/emext_scored.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print('WROTE', RD + '/emext_scored.json')
