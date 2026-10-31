#!/usr/bin/env python3
"""Free-running re-ranking of enzyme-reaction candidate pools. CPU only, no model calls.

FREE-RUNNING means nothing about the target row may be used. Every feature is computed from
(a) the candidate string, (b) the training corpus, (c) the enzyme identity. In particular the
reactant side comes from the CANDIDATE, never from the true reaction -- that is the difference
from rerank_conservation.py, whose 0.81% -> 10.39% product-set top-1 lift used the TRUE reactant
composition and therefore belongs to the reactants-given setting, not to free-running generation.

subcommands
  synth     build a synthetic pool from real reactions (unit test; correct behaviour known a priori)
  features  pool CSV(s) -> one row per (eval row, distinct candidate key) with all ranking features
  score     feature table -> policy comparison (optionally fitting the learned ranker on another table)

Policies: (1) self-balance, (2) novelty filter, (3) draw frequency, (4) learned ranker, and combinations.
Target label is the row's OWN reaction unless --target says otherwise; labels are never features.
"""
import argparse, collections, csv, json, math, os, random, sys, time
from math import sqrt

MAXTOK = 2000
KS = [1, 5, 10, 25, 50, 100, 200, 400]
T0 = time.time()
_MISS = object()
_tok = {}

FEATS = ['freq', 'logfreq', 'best_rank', 'bal_heavy', 'bal_withH', 'bal_charge', 'is_identity',
         'frac_shared_tok', 'n_react_tok', 'n_prod_tok', 'n_heavy_r', 'n_heavy_p', 'd_heavy',
         'in_train_key', 'in_trainval_key', 'prod_set_in_train', 'react_set_in_train',
         'prod_pop', 'react_pop', 'n_tok_untrained', 'min_tok_freq', 'mean_tok_freq',
         'tanimoto_rp', 'enz_n_train', 'enz_in_train']


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


def _rd(emd, s):
    with open(os.path.join(emd, 'em_ext_%s.clean.csv' % s), newline='') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- chemistry
def tokinfo(tk):
    r = _tok.get(tk, _MISS)
    if r is not _MISS:
        return r
    from rdkit import Chem
    out = None
    if tk and 'UNK' not in tk and len(tk) <= MAXTOK:
        m = Chem.MolFromSmiles(tk)
        if m is not None:
            cnt = collections.Counter(); chg = 0
            for a in m.GetAtoms():
                cnt[a.GetSymbol()] += 1
                cnt['H'] += a.GetTotalNumHs()
                chg += a.GetFormalCharge()
            out = (Chem.MolToSmiles(m, isomericSmiles=True), Chem.MolToSmiles(m, isomericSmiles=False),
                   cnt, chg, m.GetNumHeavyAtoms())
    _tok[tk] = out
    return out


def side(s):
    if not s:
        return None
    iso, flat, cnt, chg, hv = [], [], collections.Counter(), 0, 0
    for tk in s.split('.'):
        t = tokinfo(tk.strip())
        if t is None:
            return None
        iso.append(t[0]); flat.append(t[1]); cnt.update(t[2]); chg += t[3]; hv += t[4]
    return (tuple(sorted(iso)), tuple(sorted(flat)), cnt, chg, hv)


def parse(rxn):
    """Requires exactly one '>>' -- a candidate with 0 or 2+ arrows is not a reaction."""
    if not isinstance(rxn, str):
        return None
    p = rxn.split('>>')
    if len(p) != 2:
        return None
    l, r = side(p[0]), side(p[1])
    if l is None or r is None:
        return None
    return dict(key=(l[0], r[0]), L=l, R=r)


def l1(a, b, skipH=False):
    ks = set(a) | set(b)
    if skipH:
        ks.discard('H')
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in ks)


def wilson(h, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = h / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - m), 100 * (c + m)


# ---------------------------------------------------------------- corpus
class Corpus:
    def __init__(self, emd):
        from rdkit.Chem import rdFingerprintGenerator
        self.enz_known = collections.defaultdict(dict)
        self.key = {}
        self.tokfreq = collections.Counter(); self.prod_cnt = collections.Counter()
        self.react_cnt = collections.Counter(); self.enz_train_n = collections.Counter()
        self.train_tokens = set()
        for s in ('train', 'val', 'test'):
            ks = set()
            for r in _rd(emd, s):
                p = parse(r['rxnsmiles_ordered'])
                if p is None:
                    continue
                ks.add(p['key'])
                self.enz_known[r['Entry']].setdefault(p['key'], s)
                if s == 'train':
                    self.prod_cnt[p['key'][1]] += 1; self.react_cnt[p['key'][0]] += 1
                    self.enz_train_n[r['Entry']] += 1
                    for t in set(p['key'][0] + p['key'][1]):
                        self.tokfreq[t] += 1; self.train_tokens.add(t)
            self.key[s] = ks
        self.train_val = self.key['train'] | self.key['val']
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        self._fp = {}

    def fp(self, tok):
        from rdkit import Chem
        f = self._fp.get(tok, _MISS)
        if f is _MISS:
            m = Chem.MolFromSmiles(tok)
            f = self.gen.GetFingerprint(m) if m is not None else None
            self._fp[tok] = f
        return f

    def side_fp(self, toks):
        fps = [f for f in (self.fp(t) for t in toks) if f is not None]
        if not fps:
            return None
        acc = fps[0]
        for f in fps[1:]:
            acc = acc | f
        return acc


# ---------------------------------------------------------------- features
def emit(j, acc, corp, targets, bp, out):
    from rdkit import DataStructs
    for key, e in acc.items():
        L, R = e['p']['L'], e['p']['R']
        toks = set(key[0] + key[1])
        tf = [corp.tokfreq.get(t, 0) for t in toks]
        fr, fpd = corp.side_fp(key[0]), corp.side_fp(key[1])
        tan = DataStructs.TanimotoSimilarity(fr, fpd) if (fr is not None and fpd is not None) else -1.0
        shared = len(set(key[0]) & set(key[1]))
        out.append(dict(
            row_idx=j, entry=e['ent'], hit=int(key in targets),
            prod_hit=int(any(key[1] == t[1] for t in targets)),
            freq=e['freq'], logfreq=round(math.log1p(e['freq']), 4), best_rank=e['best'],
            bal_heavy=l1(L[2], R[2], skipH=True), bal_withH=l1(L[2], R[2]), bal_charge=abs(L[3] - R[3]),
            is_identity=int(key[0] == key[1]),
            frac_shared_tok=round(shared / max(1, len(set(key[0]) | set(key[1]))), 4),
            n_react_tok=len(key[0]), n_prod_tok=len(key[1]), n_heavy_r=L[4], n_heavy_p=R[4],
            d_heavy=abs(L[4] - R[4]),
            in_train_key=int(key in corp.key['train']), in_trainval_key=int(key in corp.train_val),
            prod_set_in_train=int(key[1] in corp.prod_cnt), react_set_in_train=int(key[0] in corp.react_cnt),
            prod_pop=corp.prod_cnt.get(key[1], 0), react_pop=corp.react_cnt.get(key[0], 0),
            n_tok_untrained=sum(1 for t in toks if t not in corp.train_tokens),
            min_tok_freq=min(tf) if tf else 0, mean_tok_freq=round(sum(tf) / max(1, len(tf)), 2),
            tanimoto_rp=round(tan, 4), enz_n_train=corp.enz_train_n.get(e['ent'], 0),
            enz_in_train=int(corp.enz_train_n.get(e['ent'], 0) > 0),
            blast_pool_hit=(int(key[1] in bp) if bp is not None else -1)))


def cmd_features(a):
    corp = Corpus(a.emd)
    ev = list(csv.DictReader(open(a.eval_csv, newline='')))
    ent = [r['Entry'] for r in ev]
    truth = [(parse(r['rxnsmiles_ordered']) or {}).get('key') for r in ev]
    blast = None
    if a.blast and os.path.exists(a.blast):
        blast = {k: set(tuple(tuple(y) for y in x) for x in v) for k, v in json.load(open(a.blast)).items()}
    log('corpus train keys', len(corp.key['train']), 'enzymes', len(corp.enz_known), 'eval rows', len(ev))
    st = collections.Counter()
    out, w, fh = [], None, open(a.out, 'w', newline='')

    def flush(j, acc):
        if not acc:
            return
        e = ent[j]
        known = corp.enz_known.get(e, {})
        if a.target == 'own':
            tg = {truth[j]} if truth[j] else set()
        else:
            tg = {k for k, s in known.items() if s == a.target.split('_')[1]}
        buf = []
        emit(j, acc, corp, tg, (blast or {}).get(e), buf)
        nonlocal w
        if w is None:
            w = csv.DictWriter(fh, fieldnames=list(buf[0].keys())); w.writeheader()
        w.writerows(buf); st['feat_rows'] += len(buf); st['rows'] += 1

    for path in a.pools:
        with open(path, newline='') as f:
            rd = csv.reader(f); hdr = next(rd)
            ix = {h: i for i, h in enumerate(hdr)}
            ri, pi, ki = ix['row_idx'], ix['Pred SMILES'], ix.get('rank')
            cur, acc, n = None, {}, 0
            for rec in rd:
                j = int(rec[ri]); rank = int(rec[ki]) if ki is not None else n
                if j != cur:
                    if cur is not None:
                        flush(cur, acc)
                    cur, acc, n = j, {}, 0
                n += 1
                if a.limit_rank is not None and rank >= a.limit_rank:
                    continue
                st['cand'] += 1
                p = parse(rec[pi])
                if p is None:
                    st['unparseable'] += 1; continue
                e = acc.get(p['key'])
                if e is None:
                    acc[p['key']] = dict(p=p, freq=1, best=rank, ent=ent[j])
                else:
                    e['freq'] += 1
                    e['best'] = min(e['best'], rank)
            if cur is not None:
                flush(cur, acc)
    fh.close()
    log('wrote', a.out, dict(st))


# ---------------------------------------------------------------- policies
def P(filt, keys):
    return lambda d: (d[filt(d)] if filt else d, keys)


NOV = lambda d: (d.in_trainval_key == 0) & (d.is_identity == 0)
IDENT_ONLY = lambda d: d.is_identity == 0
BAL0 = lambda d: (d.bal_heavy == 0) & (d.in_trainval_key == 0) & (d.is_identity == 0)
POLICIES = [
    ('draw order (baseline)', P(None, ['best_rank'])),
    ('self-balance heavy (1)', P(None, ['bal_heavy', 'best_rank'])),
    ('self-balance withH (1)', P(None, ['bal_withH', 'best_rank'])),
    ('novelty filter (2)', P(NOV, ['best_rank'])),
    ('draw frequency (3)', P(None, ['_negfreq', 'best_rank'])),
    ('novelty+frequency (2+3)', P(NOV, ['_negfreq', 'best_rank'])),
    ('novelty+balance (1+2)', P(NOV, ['bal_heavy', 'best_rank'])),
    ('novelty+balance+freq (1+2+3)', P(NOV, ['bal_heavy', '_negfreq', 'best_rank'])),
    # balanced-ONLY variants: filter instead of sort, so every ranked candidate is mass-balanced.
    # These truncate (mean 44.6 balanced candidates/row), which is the point of the comparison.
    ('novelty+balance+freq BALANCED-ONLY', P(lambda d: (d.in_trainval_key == 0) & (d.is_identity == 0) & (d.bal_heavy == 0), ['_negfreq', 'best_rank'])),
    ('identity+balance+freq BALANCED-ONLY (DEPLOYABLE)', P(lambda d: (d.is_identity == 0) & (d.bal_heavy == 0), ['_negfreq', 'best_rank'])),
    ('balanced-only+freq', P(BAL0, ['_negfreq', 'best_rank'])),
    # deployment-legitimate arms: NO novelty filter. The novelty filter drops candidates equal to
    # a known train/val reaction, which is sound on a reaction-disjoint benchmark but assumes the
    # answer is absent from the database -- an assumption a deployed system cannot make. These two
    # arms quantify what the re-ranker achieves without it.
    # DEPLOYMENT-LEGITIMATE arms. NOV bundles two conditions: in_trainval_key==0 (a lookup
    # against the training database, which assumes the answer is not an already-known reaction
    # -- true on this reaction-disjoint benchmark, not assumable in deployment) and
    # is_identity==0 (reactants==products, a self-contained sanity rule that needs no database).
    # The earlier "NO novelty" arms dropped BOTH, which understates deployable performance by
    # letting identity reactions occupy the top ranks. These keep the identity filter and drop
    # only the database lookup, which is the honest deployment number.
    ('identity-filter + balance + freq (DEPLOYABLE)', P(IDENT_ONLY, ['bal_heavy', '_negfreq', 'best_rank'])),
    ('identity-filter + balanced-only + freq (DEPLOYABLE)', P(lambda d: (d.is_identity == 0) & (d.bal_heavy == 0), ['_negfreq', 'best_rank'])),
    ('identity-filter only', P(IDENT_ONLY, ['best_rank'])),
    ('balance+freq, NO novelty (1+3)', P(None, ['bal_heavy', '_negfreq', 'best_rank'])),
    ('balanced-only+freq, NO novelty', P(lambda d: d.bal_heavy == 0, ['_negfreq', 'best_rank'])),
    ('learned (4)', P(None, ['_negscore'])),
    ('learned+novelty (2+4)', P(NOV, ['_negscore'])),
]


def cmd_score(a):
    import numpy as np, pandas as pd
    te = pd.read_csv(a.feats); te['_negfreq'] = -te.freq
    info, pol = {}, list(POLICIES)
    if a.train:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold, cross_val_score
        tr = pd.read_csv(a.train)
        X, y, g = tr[FEATS].values, tr.hit.values, tr.entry.values
        info.update(train_rows=int(len(tr)), train_pos=int(y.sum()), train_enzymes=int(len(set(g))))
        if y.sum() >= 20:
            mk = lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.08,
                                                        class_weight='balanced', random_state=0)
            try:
                cv = cross_val_score(mk(), X, y, groups=g, cv=GroupKFold(n_splits=4), scoring='average_precision')
                info['cv_average_precision'] = [round(float(v), 4) for v in cv]
            except Exception as e:
                info['cv_error'] = str(e)[:150]
            m = mk(); m.fit(X, y)
            te['_negscore'] = -m.predict_proba(te[FEATS].values)[:, 1]
        else:
            info['learned_skipped'] = 'too few positives'
    if '_negscore' not in te.columns:
        pol = [p for p in pol if 'learned' not in p[0]]
    rows = []
    for name, fn in pol:
        pf, pp = {}, {}
        for j, d in te.groupby('row_idx', sort=False):
            sel, keys = fn(d)
            if not len(sel):
                pf[j] = pp[j] = -1; continue
            s = sel.sort_values(keys, kind='mergesort')
            for lab, dst in (('hit', pf), ('prod_hit', pp)):
                nz = np.flatnonzero(s[lab].values)
                dst[j] = int(nz[0]) if len(nz) else -1
        for K in KS:
            hf = sum(1 for v in pf.values() if 0 <= v < K)
            hp = sum(1 for v in pp.values() if 0 <= v < K)
            lo, hi = wilson(hf, a.n_rows)
            rows.append(dict(policy=name, K=K, full_hits=hf, full_pct=round(100 * hf / a.n_rows, 3),
                             full_lo=round(lo, 2), full_hi=round(hi, 2), prod_hits=hp,
                             prod_pct=round(100 * hp / a.n_rows, 3), n_rows=a.n_rows))
    T = pd.DataFrame(rows); T.to_csv(a.prefix + '_policies.csv', index=False)
    if a.bucket_csv:
        bk = pd.read_csv(a.bucket_csv)[['row_idx', 'bucket']]
        bmap = dict(zip(bk.row_idx, bk.bucket))
        te['_bucket'] = te.row_idx.map(bmap)
        brows = []
        for bname, bsub in te.groupby('_bucket'):
            nb = int(bk[bk.bucket == bname].row_idx.nunique())
            for name, fn in pol:
                pf, pp = {}, {}
                for j, d in bsub.groupby('row_idx', sort=False):
                    sel, keys = fn(d)
                    if not len(sel):
                        pf[j] = pp[j] = -1; continue
                    sx = sel.sort_values(keys, kind='mergesort')
                    for lab, dst in (('hit', pf), ('prod_hit', pp)):
                        nz = np.flatnonzero(sx[lab].values)
                        dst[j] = int(nz[0]) if len(nz) else -1
                for K in KS:
                    hf = sum(1 for v in pf.values() if 0 <= v < K)
                    hp = sum(1 for v in pp.values() if 0 <= v < K)
                    lo, hi = wilson(hf, nb)
                    brows.append(dict(bucket=bname, policy=name, K=K, full_hits=hf,
                                      full_pct=round(100 * hf / nb, 3), full_lo=round(lo, 2),
                                      full_hi=round(hi, 2), prod_hits=hp,
                                      prod_pct=round(100 * hp / nb, 3), n_rows=nb))
        pd.DataFrame(brows).to_csv(a.prefix + '_by_bucket.csv', index=False)
    info.update(oracle_full=int(te.groupby('row_idx').hit.max().sum()),
                oracle_prod=int(te.groupby('row_idx').prod_hit.max().sum()),
                rows_with_candidates=int(te.row_idx.nunique()), n_rows=a.n_rows,
                mean_distinct=round(float(te.groupby('row_idx').size().mean()), 1))
    json.dump(info, open(a.prefix + '_info.json', 'w'), indent=1)
    pd.set_option('display.width', 220)
    print(T[T.K.isin([1, 5, 10])].pivot_table(index='policy', columns='K', values='full_hits').to_string())
    print('INFO', json.dumps(info))


# ---------------------------------------------------------------- synthetic unit test
def cmd_synth(a):
    rnd = random.Random(a.seed)
    tr, te = _rd(a.emd, 'train'), _rd(a.emd, 'test')
    by = {}
    for r in tr:
        by.setdefault(r['Entry'], []).append(r['rxnsmiles_ordered'])
    sel = [i for i, r in enumerate(te) if r['Entry'] in by][:a.n_rows]
    allrx = [r['rxnsmiles_ordered'] for r in tr]
    planted = {}
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['row_idx', 'rank', 'True SMILES', 'Pred SMILES'])
        for j in sel:
            truth = te[j]['rxnsmiles_ordered']; pool = []
            has = rnd.random() < a.hit_frac
            if has:
                pool += [truth] * a.true_dup                      # the answer, rare
            for t in by[te[j]['Entry']][:3]:
                pool += [t] * 6                                   # memorised train reactions, frequent
            for _ in range(3):
                o = rnd.choice(allrx); l, r = o.split('>>', 1); ps = r.split('.')
                pool.append(l + '>>' + ('.'.join(ps[:-1]) if len(ps) > 1 else r))   # unbalanced
            l = truth.split('>>')[0]
            pool.append(l + '>>' + l)                             # identity
            pool.append('not a reaction at all')                  # malformed
            pool += [rnd.choice(allrx) for _ in range(4)]         # unrelated real reactions
            rnd.shuffle(pool); planted[j] = has
            for k, smi in enumerate(pool):
                w.writerow([j, k, truth, smi])
    print(json.dumps(dict(rows=len(sel), rows_with_truth=sum(planted.values()), out=a.out)))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    EMD = '/path/to/struct_rxnpred/data/em_ext'
    f = sub.add_parser('features'); f.add_argument('out'); f.add_argument('pools', nargs='+')
    f.add_argument('--eval-csv', required=True); f.add_argument('--emd', default=EMD)
    f.add_argument('--target', default='own', choices=['own', 'enzyme_val', 'enzyme_test'])
    f.add_argument('--blast', default=None); f.add_argument('--limit-rank', type=int, default=None)
    f.set_defaults(fn=cmd_features)
    s = sub.add_parser('score'); s.add_argument('prefix'); s.add_argument('feats')
    s.add_argument('--train', default=None); s.add_argument('--n-rows', type=int, required=True)
    s.add_argument('--bucket-csv', default=None,
                   help='CSV with row_idx,bucket (e.g. audit/full_rows.csv) for stratified output')
    s.set_defaults(fn=cmd_score)
    y = sub.add_parser('synth'); y.add_argument('out'); y.add_argument('--emd', default=EMD)
    y.add_argument('--n-rows', type=int, default=120); y.add_argument('--hit-frac', type=float, default=0.6)
    y.add_argument('--true-dup', type=int, default=2); y.add_argument('--seed', type=int, default=0)
    y.set_defaults(fn=cmd_synth)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
