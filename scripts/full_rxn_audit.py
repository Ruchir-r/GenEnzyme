#!/usr/bin/env python3
"""Full-reaction + novelty audit of em_ext candidate pools (CPU only, no model calls).

usage: full_rxn_audit.py OUTPREFIX MODE name=poolfile [name=poolfile ...]
  MODE = annotate  -> also write OUTPREFIX_rows.csv / OUTPREFIX_meta.json (per-test-row annotations)
  MODE = noannot   -> pools only

For every pool (rows of: row_idx, rank, True SMILES, Pred SMILES) it records, per test row, the
FIRST RANK at which each criterion is met (-1 = never), so recall@K for any K <= pool size is
(first_rank >= 0 and first_rank < K):
  prod_strict   product multiset, isomeric canonical SMILES  (the metric used so far)
  prod_const    product multiset, stereo removed
  react_strict  generated REACTANT multiset equals the true reactant multiset
  full_strict   exactly one '>>', reactants AND products both strict-equal
  full_const    same, stereo removed
  main_strict   product multiset after deleting the TOP_COPROD most frequent training metabolites
                (water, proton, nucleotide cofactors ...) from both sides
"""
import csv, sys, json, re, collections, time
from multiprocessing import Pool
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10 ** 8)

R2 = '/path/to/struct_rxnpred'
EMD = R2 + '/data/em_ext'
FE = R2 + '/claude_dir/17_reaction_revival/final_eval'
H18 = R2 + '/claude_dir/18_horizyn/blast_baseline'
MAXTOK = 2000
TOP_COPROD = 15
_MISSING = object()
_tok = {}
_par = {}
_unch = rdMolStandardize.Uncharger()


def tokinfo(tk):
    r = _tok.get(tk, _MISSING)
    if r is not _MISSING:
        return r
    out = None
    if tk and 'UNK' not in tk and len(tk) <= MAXTOK:
        m = Chem.MolFromSmiles(tk)
        if m is not None:
            out = (Chem.MolToSmiles(m, isomericSmiles=True), Chem.MolToSmiles(m, isomericSmiles=False))
    _tok[tk] = out
    return out


def parent(tk):
    r = _par.get(tk, _MISSING)
    if r is not _MISSING:
        return r
    out = None
    m = Chem.MolFromSmiles(tk) if (tk and len(tk) <= MAXTOK and 'UNK' not in tk) else None
    if m is not None:
        try:
            m = _unch.uncharge(m)
        except Exception:
            pass
        out = Chem.MolToSmiles(m, isomericSmiles=False)
    _par[tk] = out
    return out


def side(s):
    """(iso_tuple, flat_tuple) of sorted canonical tokens, or None if any token is unparseable."""
    if not s:
        return None
    iso, flat = [], []
    for tk in s.split('.'):
        t = tokinfo(tk.strip())
        if t is None:
            return None
        iso.append(t[0]); flat.append(t[1])
    return (tuple(sorted(iso)), tuple(sorted(flat)))


TRUTH = {}      # row_idx -> (react_side, prod_side, main_true)
TESTRX = []
COPROD = set()


def process_pool(task):
    name, path = task
    t0 = time.time()
    first = {}
    ncand = collections.Counter()
    uniq = {}
    st = collections.Counter()
    with open(path, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd)
        ix = {h: i for i, h in enumerate(hdr)}
        ri, pi, ti = ix['row_idx'], ix['Pred SMILES'], ix.get('True SMILES')
        ki = ix.get('rank')
        for row in rd:
            r = int(row[ri])
            rank = int(row[ki]) if ki is not None else ncand[r]
            ncand[r] += 1
            st['cand'] += 1
            if ti is not None and row[ti] != TESTRX[r]:
                st['align_mismatch'] += 1
            tr = TRUTH.get(r)
            fh = first.get(r)
            if fh is None:
                fh = first[r] = [-1] * 6
            parts = row[pi].split('>>')
            if len(parts) < 2:
                st['no_arrow'] += 1
                continue
            well = (len(parts) == 2)
            if well:
                st['well'] += 1
            prod = side(parts[-1])
            reac = side(parts[0])
            if prod is None:
                st['prod_invalid'] += 1
            else:
                uniq.setdefault(r, set()).add(prod[0])
            if tr is None:
                continue
            tre, trp, trm = tr
            hits = [False] * 6
            if prod is not None and trp is not None:
                hits[0] = prod[0] == trp[0]
                hits[1] = prod[1] == trp[1]
                if trm:
                    mp = tuple(t for t in prod[0] if t not in COPROD)
                    hits[5] = (mp == trm)
            if reac is not None and tre is not None:
                hits[2] = reac[0] == tre[0]
                if well and prod is not None and trp is not None:
                    hits[3] = hits[2] and hits[0]
                    hits[4] = (reac[1] == tre[1]) and hits[1]
            for k in range(6):
                if hits[k] and (fh[k] < 0 or rank < fh[k]):
                    fh[k] = rank
    rows = []
    for r in sorted(first):
        rows.append([name, r, ncand[r], len(uniq.get(r, ())) ] + first[r])
    st['seconds'] = round(time.time() - t0, 1)
    return name, rows, dict(st)


def annotate(prefix):
    global COPROD
    test = list(csv.DictReader(open(EMD + '/em_ext_test.clean.csv')))
    train = list(csv.DictReader(open(EMD + '/em_ext_train.clean.csv')))
    TESTRX.extend(r['rxnsmiles_ordered'] for r in test)
    raw_train = set(); canon_train = set(); parent_train = set()
    tokfreq = collections.Counter(); keys = set(); react_sets = set(); prod_cnt = collections.Counter()
    train_entries = set(r['Entry'] for r in train)
    bad_train = 0
    for r in train:
        rx = r['rxnsmiles_ordered']; parts = rx.split('>>')
        e, p = side(parts[0]), side(parts[-1])
        if e is None or p is None:
            bad_train += 1
            continue
        toks = [t for t in re.split(r'>>|\.', rx) if t]
        raw_train.update(toks)
        for t in set(e[0] + p[0]):
            tokfreq[t] += 1
        canon_train.update(e[0]); canon_train.update(p[0])
        keys.add((e[0], p[0])); react_sets.add(e[0]); prod_cnt[p[0]] += 1
    for t in raw_train:
        pk = parent(t)
        if pk:
            parent_train.add(pk)
    COPROD = set(t for t, _ in tokfreq.most_common(TOP_COPROD))
    pop_rank = {k: i for i, (k, _) in enumerate(prod_cnt.most_common())}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tr_list = sorted(canon_train)
    tr_fps = [gen.GetFingerprint(Chem.MolFromSmiles(t)) for t in tr_list]
    nn_cache = {}

    def nn_tan(tok):
        if tok not in nn_cache:
            fp = gen.GetFingerprint(Chem.MolFromSmiles(tok))
            nn_cache[tok] = max(DataStructs.BulkTanimotoSimilarity(fp, tr_fps))
        return nn_cache[tok]

    # BLAST: best non-self hit per query by bitscore
    qlen = {}
    cur = None
    for line in open(H18 + '/test_query.fasta'):
        if line.startswith('>'):
            cur = line[1:].split()[0]; qlen[cur] = 0
        else:
            qlen[cur] += len(line.strip())
    best = {}; selfhit = set()
    for line in open(H18 + '/blastp_hits.tsv'):
        f = line.rstrip('\n').split('\t')
        q, s = f[0], f[1]
        if q == s:
            selfhit.add(q); continue
        bits = float(f[11])
        if q not in best or bits > best[q][0]:
            best[q] = (bits, float(f[2]), (int(f[7]) - int(f[6]) + 1) / max(qlen.get(q, 1), 1))
    ent_count = collections.Counter(r['Entry'] for r in test)
    rows_out = []
    meta = collections.Counter()
    for i, r in enumerate(test):
        rx = r['rxnsmiles_ordered']; parts = rx.split('>>')
        e, p = side(parts[0]), side(parts[-1])
        if e is None or p is None:
            TRUTH[i] = None; meta['unparseable_true'] += 1
            continue
        main = tuple(t for t in p[0] if t not in COPROD)
        TRUTH[i] = (e, p, main)
        raw_l = [t for t in parts[0].split('.') if t]; raw_r = [t for t in parts[-1].split('.') if t]
        exact_seen = (e[0], p[0]) in keys
        all_seen = all(t in canon_train for t in e[0] + p[0])
        prod_unseen = [t for t in p[0] if t not in canon_train]
        react_unseen = [t for t in e[0] if t not in canon_train]
        if exact_seen:
            bucket = 'exact_seen'
        elif all_seen:
            bucket = 'recombination'
        else:
            bucket = 'novel_metabolite'
        cat = 'recombination' if bucket != 'novel_metabolite' else ('novel_reactant_only' if not prod_unseen else 'novel_product')
        prod_untrained_raw = any(t not in raw_train for t in raw_r)
        pp = [parent(t) for t in raw_r]; rp = [parent(t) for t in raw_l]
        prod_parent_unseen = any((x is not None and x not in parent_train) for x in pp)
        any_parent_unseen = prod_parent_unseen or any((x is not None and x not in parent_train) for x in rp)
        nn = min(nn_tan(t) for t in prod_unseen) if prod_unseen else ''
        ent = r['Entry']
        b = best.get(ent)
        meta[bucket] += 1; meta[cat] += 1
        rows_out.append(dict(row_idx=i, entry=ent, enzyme_in_train=int(ent in train_entries), n_rows_enzyme=ent_count[ent],
            bucket=bucket, cat=cat, prod_untrained_raw=int(prod_untrained_raw), prod_parent_unseen=int(prod_parent_unseen),
            any_parent_unseen=int(any_parent_unseen), prod_set_seen=int(p[0] in prod_cnt), react_set_seen=int(e[0] in react_sets),
            pop_count=prod_cnt.get(p[0], 0), pop_rank=pop_rank.get(p[0], -1), nn_tan_unseen_prod=nn,
            n_prod_tok=len(p[0]), main_defined=int(bool(main)), blast_self=int(ent in selfhit),
            blast_best_pident=(b[1] if b else ''), blast_best_qcov=(round(b[2], 3) if b else ''), blast_best_bits=(b[0] if b else '')))
    with open(prefix + '_rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    meta = dict(meta)
    meta.update(n_test=len(test), n_train=len(train), bad_train=bad_train, coprod=[[t, tokfreq[t]] for t in sorted(COPROD, key=lambda x: -tokfreq[x])],
                n_train_tokens_raw=len(raw_train), n_train_tokens_canon=len(canon_train), n_train_prod_sets=len(prod_cnt),
                n_train_react_sets=len(react_sets), pop_top5=[[list(k), v] for k, v in prod_cnt.most_common(5)])
    json.dump(meta, open(prefix + '_meta.json', 'w'), indent=1)
    print('ANNOTATED', json.dumps({k: v for k, v in meta.items() if k not in ('coprod', 'pop_top5')}), flush=True)
    print('COPROD', meta['coprod'], flush=True)


def main():
    prefix, mode = sys.argv[1], sys.argv[2]
    pools = [a.split('=', 1) for a in sys.argv[3:]]
    if mode == 'annotate':
        annotate(prefix)
    else:
        # truth rows + co-product set must be rebuilt: reuse annotate() (writes a throwaway *_tmp_rows.csv)
        annotate(prefix + '_tmp')
    tasks = [(n, p if p.startswith('/') else FE + '/' + p) for n, p in pools]
    out = []
    stats = {}
    with Pool(min(8, max(1, len(tasks)))) as pool:
        for name, rows, st in pool.imap_unordered(process_pool, tasks):
            out.extend(rows); stats[name] = st
            n_hit = sum(1 for r in rows if r[4] >= 0)
            print('POOL', name, 'rows', len(rows), 'prod_strict_hits', n_hit, 'prod_const_hits', sum(1 for r in rows if r[5] >= 0),
                  'full_strict_hits', sum(1 for r in rows if r[7] >= 0), json.dumps(st), flush=True)
    out.sort(key=lambda x: (x[0], x[1]))
    with open(prefix + '_hits.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool', 'row_idx', 'n_cand', 'n_unique_prod', 'fh_prod_strict', 'fh_prod_const', 'fh_react_strict', 'fh_full_strict', 'fh_full_const', 'fh_main_strict'])
        w.writerows(out)
    json.dump(stats, open(prefix + '_poolstats.json', 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

