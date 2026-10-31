#!/usr/bin/env python3
"""Enzyme-level recovery of KNOWN reactions (train + val + test) by free-running candidate pools. CPU only.

usage: enzyme_known_recall.py OUTPREFIX test=POOL.csv [A=POOL_A.csv B=POOL_B.csv]

Definition ("correct" under the new rule): a generated candidate is correct if its canonical reaction key
(sorted reactant multiset >> sorted product multiset, isomeric SMILES) equals ANY known reaction of the
enzyme it was generated for, in ANY split (train, val or test). Because the model input is the enzyme alone
(graph + ESM embedding; no reactants, no id-override), one pool per enzyme is enough.

  test pool  : row_idx indexes em_ext_test.clean.csv; all 2,349 rows are scored (row level) and the pool of
               the FIRST test row of each test enzyme is its enzyme pool.
  shard pools: row_idx indexes enz_pool_{A,B}.clean.csv (one row per non-test enzyme).
Candidate rank = draw index (first N draws = an N-draw run).

Outputs (OUTPREFIX_*): known_pairs.csv, row_level.csv, global_counts.csv, perm_control.csv,
enzyme_pool_summary.csv, diag.json
"""
import csv, sys, json, collections, random, time, os
D17 = '/path/to/struct_rxnpred/claude_dir/17_reaction_revival'
R2 = '/path/to/struct_rxnpred'
sys.path.insert(0, D17)
from full_rxn_audit import side
csv.field_size_limit(10 ** 8)
NS = [1, 5, 10, 25, 50, 100, 200, 400]
INF = 10 ** 9
T0 = time.time()


def log(*a):
    print('[%.0fs]' % (time.time() - T0), *a, flush=True)


def rkey(rx):
    p = rx.split('>>')
    if len(p) != 2:
        return None
    a, b = side(p[0]), side(p[1])
    if a is None or b is None:
        return None
    return ((a[0], b[0]), (a[1], b[1]))          # (iso key, constitutional key)


def load(path):
    return list(csv.DictReader(open(path, newline='')))


def stream(path):
    seen = set()
    with open(path, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ix = {h: i for i, h in enumerate(hdr)}
        ri, pi, ti, ki = ix['row_idx'], ix['Pred SMILES'], ix.get('True SMILES'), ix.get('rank')
        cur, cand, truth = None, [], None
        for row in rd:
            r = int(row[ri])
            if r != cur:
                if cur is not None:
                    yield cur, truth, cand
                    seen.add(cur)
                assert r not in seen, 'pool rows not contiguous'
                cur, cand, truth = r, [], (row[ti] if ti is not None else None)
            cand.append((int(row[ki]) if ki is not None else len(cand), row[pi]))
        if cur is not None:
            yield cur, truth, cand


def first_ranks(cand):
    fi, ff, nwell = {}, {}, 0
    for rank, smi in cand:
        k = rkey(smi)
        if k is None:
            continue
        nwell += 1
        iso, flat = k
        if iso not in fi or rank < fi[iso]:
            fi[iso] = rank
        if flat not in ff or rank < ff[flat]:
            ff[flat] = rank
    return fi, ff, nwell


def main():
    prefix = sys.argv[1]
    pools = dict(a.split('=', 1) for a in sys.argv[2:])
    rows = {s: load('%s/data/em_ext/em_ext_%s.clean.csv' % (R2, s)) for s in ('train', 'val', 'test')}
    known = collections.defaultdict(dict)
    gk = {}
    conflicts = 0
    bad = collections.Counter()
    for s in ('train', 'val', 'test'):
        for r in rows[s]:
            k = rkey(r['rxnsmiles_ordered'])
            if k is None:
                bad[s] += 1
                continue
            e = r['Entry']
            iso, flat = k
            d = known[e].get(iso)
            if d is None:
                known[e][iso] = dict(split=s, flat=flat, rxn=r['rxnsmiles_ordered'], nrows=1)
            else:
                d['nrows'] += 1
                if d['split'] != s:
                    conflicts += 1
            prev = gk.get(iso)
            if prev is None:
                gk[iso] = s
            elif prev != s:
                conflicts += 1
    log('known reactions: enzymes', len(known), 'pairs', sum(len(v) for v in known.values()), 'distinct keys', len(gk), 'unparseable', dict(bad), 'split conflicts', conflicts)

    test_rows = rows['test']
    test_key = [rkey(r['rxnsmiles_ordered']) for r in test_rows]
    D_iso, D_flat, summ = {}, {}, {}
    row_out = []
    consist = {}          # enzyme -> list of (key set, n_found_400) for first two rows
    align_bad = collections.Counter()

    def process(src, path, entry_of, truth_of):
        n = 0
        for j, truth, cand in stream(path):
            e = entry_of[j]
            if truth is not None and truth != truth_of[j]:
                align_bad[src] += 1
            fi, ff, nwell = first_ranks(cand)
            K = known.get(e, {})
            if src == 'test':
                kj = test_key[j]
                own = fi.get(kj[0], -1) if kj else -1
                def mn(cond):
                    b = INF
                    for iso, d in K.items():
                        if cond(d['split']):
                            r = fi.get(iso)
                            if r is not None and r < b:
                                b = r
                    return b if b < INF else -1
                bc = INF
                for iso, d in K.items():
                    r = ff.get(d['flat'])
                    if r is not None and r < bc:
                        bc = r
                nf = sum(1 for iso in K if iso in fi)
                row_out.append([j, e, sum(1 for d in K.values() if d['split'] == 'train'), sum(1 for d in K.values() if d['split'] == 'val'),
                                sum(1 for d in K.values() if d['split'] == 'test'), own, mn(lambda s: True), mn(lambda s: s == 'train'),
                                mn(lambda s: s == 'val'), mn(lambda s: s == 'test'), (bc if bc < INF else -1), nf, len(cand), nwell, len(fi)])
                lst = consist.setdefault(e, [])
                if len(lst) < 2:
                    lst.append((set(fi), nf))
            if e not in D_iso:
                D_iso[e], D_flat[e] = fi, ff
                summ[e] = [src, j, len(cand), nwell, len(fi)]
            n += 1
        return n

    test_entry = [r['Entry'] for r in test_rows]
    test_truth = [r['rxnsmiles_ordered'] for r in test_rows]
    if 'test' in pools:
        log('test pool rows', process('test', pools['test'], test_entry, test_truth))
    for tag in ('A', 'B'):
        if tag in pools:
            sh = load('%s/claude_dir/17_reaction_revival/final_eval/enz_pool_%s.clean.csv' % (R2, tag))
            ent = [r['Entry'] for r in sh]
            tru = [r['rxnsmiles_ordered'] for r in sh]
            log('shard', tag, 'pool rows', process('shard' + tag, pools[tag], ent, tru))
    log('enzymes with a pool', len(D_iso), 'alignment mismatches', dict(align_bad))

    # ---- row level
    with open(prefix + '_row_level.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_idx', 'entry', 'nk_train', 'nk_val', 'nk_test', 'fr_own', 'fr_any', 'fr_train', 'fr_val', 'fr_test', 'fr_any_const', 'n_known_found_400', 'n_cand', 'n_well', 'n_distinct'])
        w.writerows(row_out)
    # ---- known pairs
    miss_pool = 0
    with open(prefix + '_known_pairs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['entry', 'split', 'nrows_with_key', 'n_known_enzyme', 'fr_strict', 'fr_const', 'fr_reverse', 'rxn'])
        for e, K in known.items():
            if e not in D_iso:
                miss_pool += 1
                continue
            D, Df = D_iso[e], D_flat[e]
            for iso, d in K.items():
                w.writerow([e, d['split'], d['nrows'], len(K), D.get(iso, -1), Df.get(d['flat'], -1), D.get((iso[1], iso[0]), -1), d['rxn']])
    # ---- global counts (dataset-level correctness) per enzyme pool
    cats = ['own_train', 'own_val', 'own_test', 'for_train', 'for_val', 'for_test']
    with open(prefix + '_global_counts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['entry', 'N', 'n_distinct'] + cats)
        for e, D in D_iso.items():
            K = known.get(e, {})
            rk = {c: [] for c in cats}
            for iso, r in D.items():
                s = gk.get(iso)
                if s is None:
                    continue
                rk[('own_' if iso in K else 'for_') + s].append(r)
            for N in NS:
                w.writerow([e, N, sum(1 for r in D.values() if r < N)] + [sum(1 for r in rk[c] if r < N) for c in cats])
    # ---- permutation control: pool of e vs known set of pi(e)
    ents = sorted(e for e in known if e in D_iso)
    NSP = [25, 100, 400]
    perm_rows = []

    def derange(seed):
        rnd = random.Random(seed)
        p = ents[:]
        rnd.shuffle(p)
        while True:
            fixed = [i for i in range(len(ents)) if p[i] == ents[i]]
            if not fixed:
                return p
            i = fixed[0]
            j = (i + 1) % len(ents)
            p[i], p[j] = p[j], p[i]

    def evaluate(label, mapping):
        tot = collections.Counter(); fnd = collections.Counter(); hit_enz = collections.Counter(); enz_with = collections.Counter()
        for e, f in zip(ents, mapping):
            D = D_iso[e]
            Kf = known[f]
            by = collections.defaultdict(list)
            for iso, d in Kf.items():
                by[d['split']].append(iso)
            for s, isos in by.items():
                for N in NSP:
                    tot[(s, N)] += len(isos)
                    c = sum(1 for iso in isos if D.get(iso, INF) < N)
                    fnd[(s, N)] += c
                    enz_with[(s, N)] += 1
                    hit_enz[(s, N)] += (c > 0)
        for (s, N), t in sorted(tot.items()):
            perm_rows.append([label, N, s, fnd[(s, N)], t, hit_enz[(s, N)], enz_with[(s, N)]])
    evaluate('real', ents)
    for seed in range(5):
        evaluate('perm%d' % seed, derange(seed))
    with open(prefix + '_perm_control.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'N', 'split', 'found', 'total', 'enzymes_with_hit', 'enzymes_with_known'])
        w.writerows(perm_rows)
    with open(prefix + '_enzyme_pool_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['entry', 'src', 'row', 'n_cand', 'n_well', 'n_distinct', 'nk_train', 'nk_val', 'nk_test'])
        for e, v in summ.items():
            K = known.get(e, {})
            w.writerow([e] + v + [sum(1 for d in K.values() if d['split'] == s) for s in ('train', 'val', 'test')])
    # ---- consistency of rows of the same enzyme
    multi = [(e, v) for e, v in consist.items() if len(v) == 2]
    rnd = random.Random(0)
    allk = [e for e in D_iso]
    jw, jc = [], []
    for e, v in multi:
        a, b = v[0][0], v[1][0]
        if a and b:
            jw.append(len(a & b) / len(a | b))
            o = D_iso[rnd.choice(allk)]
            os_ = set(o)
            if os_ and a:
                jc.append(len(a & os_) / len(a | os_))
    diag = dict(n_enzymes_known=len(known), n_pools=len(D_iso), enzymes_known_without_pool=miss_pool, split_conflicts=conflicts, unparseable_known=dict(bad),
                align_mismatch=dict(align_bad), n_multi_row_enzymes=len(multi), jaccard_within_enzyme=(sum(jw) / len(jw) if jw else None), jaccard_cross_enzyme=(sum(jc) / len(jc) if jc else None),
                found_400_row1=(sum(v[0][1] for _, v in multi) / max(1, len(multi))), found_400_row2=(sum(v[1][1] for _, v in multi) / max(1, len(multi))),
                runtime_s=round(time.time() - T0, 1))
    json.dump(diag, open(prefix + '_diag.json', 'w'), indent=1)
    log('DONE', json.dumps(diag))


if __name__ == '__main__':
    main()

