"""Product-only recall@k for Horizyn screen (retrieval) CSVs, scored with the
SAME canonical product-match definition used for our own generative model
(score_tf.py / score_own_freerun.py canon_set / canon_flat): does a pool
reaction ranked <=k for this protein have a PRODUCT side matching the true
product (stereo-exact = strict, stereo-stripped = constitutional)?

This differs from score_screen_v2.py's recall@k, which requires the FULL
reaction (both reactant and product side) to match -- kept here too, for
reference, since Horizyn is a whole-reaction retrieval model.

Usage: score_hz_product.py <out.json> <k1,k2,...> <test_csv> <screen_csv> [<screen_csv> ...]
"""
import csv, sys, os, json
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10**8)


def canon_set(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if '>>' in s:
        s = s.rsplit('>>', 1)[1]
    out = []
    for tok in s.split('.'):
        tok = tok.strip()
        if not tok or 'UNK' in tok:
            return None
        m = Chem.MolFromSmiles(tok)
        if m is None:
            return None
        out.append(Chem.MolToSmiles(m, isomericSmiles=True))
    if not out:
        return None
    return tuple(sorted(out))


def canon_flat(s):
    t = canon_set(s)
    if t is None:
        return None
    flat = []
    for c in t:
        m = Chem.MolFromSmiles(c)
        if m is None:
            return None
        Chem.RemoveStereochemistry(m)
        flat.append(Chem.MolToSmiles(m, isomericSmiles=False))
    return tuple(sorted(flat))


def full_rxn_key(s):
    """(reactant multiset, product multiset), stereo-exact -- for the
    reference full-reaction recall, mirrors score_screen_v2.parse_rxn."""
    if not isinstance(s, str) or '>>' not in s:
        return None
    parts = s.rsplit('>>', 1) if s.count('>>') > 1 else s.split('>>')
    if len(parts) != 2:
        return None
    left, right = parts
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
    return (out[0], out[1])


def score(screen_csv, test_rows, ks):
    base = os.path.basename(screen_csv)
    # per-protein: min rank at which a pool candidate's product matches a
    # given (strict) product key, and (const) flattened key; also full-rxn.
    prot_prod_rank = {}   # pid -> {prod_key: min_rank}
    prot_flat_rank = {}
    prot_full_rank = {}
    with open(screen_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row['protein_id']
            rk = int(row['rank'])
            smi = row['reaction_smiles']
            pk = canon_set(smi)
            if pk is not None:
                d = prot_prod_rank.setdefault(pid, {})
                if pk not in d or rk < d[pk]:
                    d[pk] = rk
                fk = canon_flat(smi)
                if fk is not None:
                    d2 = prot_flat_rank.setdefault(pid, {})
                    if fk not in d2 or rk < d2[fk]:
                        d2[fk] = rk
            fullk = full_rxn_key(smi)
            if fullk is not None:
                d3 = prot_full_rank.setdefault(pid, {})
                if fullk not in d3 or rk < d3[fullk]:
                    d3[fullk] = rk

    n_total = len(test_rows)
    n_screened = 0
    n_unparseable = 0
    hits_s = {k: 0 for k in ks}
    hits_c = {k: 0 for k in ks}
    hits_full = {k: 0 for k in ks}
    for pid, rxn in test_rows:
        truth_s = canon_set(rxn)
        truth_c = canon_flat(rxn)
        truth_full = full_rxn_key(rxn)
        if truth_s is None:
            n_unparseable += 1
            continue
        if pid not in prot_prod_rank and pid not in prot_full_rank:
            continue  # protein not screened
        n_screened += 1
        pr = prot_prod_rank.get(pid, {})
        fr = prot_flat_rank.get(pid, {})
        fullr = prot_full_rank.get(pid, {})
        r_s = pr.get(truth_s)
        r_c = fr.get(truth_c)
        r_full = fullr.get(truth_full) if truth_full is not None else None
        for k in ks:
            if r_s is not None and r_s <= k:
                hits_s[k] += 1
            if r_c is not None and r_c <= k:
                hits_c[k] += 1
            if r_full is not None and r_full <= k:
                hits_full[k] += 1

    p = lambda x, denom: round(100.0 * x / max(denom, 1), 3)
    d = dict(csv=base, n_total=n_total, n_screened=n_screened,
             n_unparseable_true=n_unparseable)
    for k in ks:
        d['strict_product_recall@%d' % k] = p(hits_s[k], n_screened)
        d['const_product_recall@%d' % k] = p(hits_c[k], n_screened)
        d['full_rxn_recall@%d' % k] = p(hits_full[k], n_screened)
        d['hits_strict_product@%d' % k] = hits_s[k]
        d['hits_const_product@%d' % k] = hits_c[k]
        d['hits_full_rxn@%d' % k] = hits_full[k]
    print('RESULT ' + json.dumps(d), flush=True)
    return d


if __name__ == '__main__':
    outp = sys.argv[1]
    ks = [int(x) for x in sys.argv[2].split(',')]
    test_csv = sys.argv[3]
    screen_csvs = sys.argv[4:]
    test_rows = []
    with open(test_csv) as f:
        for row in csv.DictReader(f):
            test_rows.append((row['Entry'], row['rxnsmiles_ordered']))
    res = []
    for sc in screen_csvs:
        if not os.path.exists(sc):
            print('MISSING ' + sc, flush=True)
            continue
        res.append(score(sc, test_rows, ks))
    with open(outp, 'w') as fh:
        json.dump(res, fh, indent=1)
    print('WROTE %s (%d rows)' % (outp, len(res)), flush=True)
