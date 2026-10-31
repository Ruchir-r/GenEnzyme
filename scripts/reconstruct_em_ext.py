#!/usr/bin/env python3
"""Reverse-engineered reconstruction of the em_ext (EnzymeMap_ext) dataset builder.

The original build script was never found. Every rule below was inferred from the shipped files and
verified against them; the report printed at the end says, per rule, how well it reproduces em_ext.

RULES (status = how it was established):
  R1 filter+join+dedup   EXACT   EnzymeMap v2 processed_reactions.csv.gz; one pair per (accession in the row's
                                  protein_refs that has an embedding, `unmapped` reaction); drop duplicate
                                  (Entry, reaction) pairs.   -> 23,451 pairs / 11,088 reactions / 6,649 enzymes.
  R2 protonation snap    ~99%     each component of the `unmapped` string is replaced by the form RxnPred's
                                  vocabulary already holds for the same molecule (identity = canonical SMILES after
                                  forced neutralisation); molecules absent from the old vocabulary keep EnzymeMap's
                                  string (EnzymeMap writes them neutral). Component ORDER is kept as in `unmapped`.
  (R2 note: Cl, F, I stay un-snapped in em_ext but other single-atom ions (Na+, Mg2+ ...) ARE snapped; a blanket
   single-atom exception was tested and made things worse, 98.96% -> 93.81%, so it is NOT part of the rule.)
  R3 vocabulary          EXACT   new = old (ids preserved) + sorted(new metabolite tokens) appended.
  R4 split               PARTIAL group = exact reaction string; 80/10/10 filled by ROW count in a shuffled
                                  order. The seed / list order was NOT recovered, so only sizes are checked.

usage: reconstruct_em_ext.py --enzymemap processed_reactions.csv.gz --em-ext-dir DIR --old-vocab vocab.json
       [--embedded-list accessions.txt]   (on the cluster: ls /path/to/scratch/embeddings | sed 's/\\.pt$//; s/_.*//')
"""
import argparse, ast, collections, csv, json, sys
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
RDLogger.DisableLog('rdApp.*')

def accs(x):                                               # identical to the project's em_join.py
    if not isinstance(x, str) or not x.strip(): return []
    s = x.strip()
    if s.startswith('['):
        try: return [str(a).strip() for a in ast.literal_eval(s) if str(a).strip()]
        except Exception: pass
    return [t.strip() for t in s.replace(';', ',').split(',') if t.strip()]

from rdkit.Chem.MolStandardize import rdMolStandardize
_UNF = rdMolStandardize.Uncharger(canonicalOrder=True, force=True)
_id = {}
def ident(t):
    """protonation-insensitive identity = canonical isomeric SMILES after FORCED neutralisation.
    (default Uncharger keeps zwitterions such as NAD+ in several charge states; InChI merges keto/enol tautomers.
     Both were tried and rejected: 74.5% and 48.6% reaction-level reproduction vs 98.96% for this key.)"""
    if t in _id: return _id[t]
    m = Chem.MolFromSmiles(t); out = None
    if m is not None:
        try: out = Chem.MolToSmiles(_UNF.uncharge(m), isomericSmiles=True)
        except Exception: out = Chem.MolToSmiles(m, isomericSmiles=True)
    _id[t] = out; return out

def sides(s):
    l, r = s.split('>>'); return [t for t in l.split('.') if t], [t for t in r.split('.') if t]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enzymemap', required=True); ap.add_argument('--em-ext-dir', required=True)
    ap.add_argument('--old-vocab', required=True); ap.add_argument('--embedded-list', default=None)
    ap.add_argument('--out-json', default='reconstruct_report.json')
    a = ap.parse_args(); R = {}
    rd = lambda s: pd.read_csv('%s/em_ext_%s.clean.csv' % (a.em_ext_dir, s))
    tr, va, te = rd('train'), rd('val'), rd('test')
    EM = pd.concat([tr.assign(split='train'), va.assign(split='val'), te.assign(split='test')], ignore_index=True)

    # ---- R1
    df = pd.read_csv(a.enzymemap, usecols=['unmapped', 'protein_refs'], low_memory=False)
    df['acc'] = df.protein_refs.map(accs)
    emb = set(open(a.embedded_list).read().split()) if a.embedded_list else set(EM.Entry)   # proxy unless a real list is given
    R['embedded_set_source'] = 'file' if a.embedded_list else 'em_ext Entry set (proxy for the embedding cache)'
    pairs = []
    for u, v in zip(df.unmapped.values, df.acc.values):
        for x in v:
            if x in emb and isinstance(u, str): pairs.append((x, u))
    P = pd.DataFrame(pairs, columns=['Entry', 'unmapped']).drop_duplicates()
    R['R1'] = dict(pairs=len(P), em_ext_rows=len(EM), reactions=int(P.unmapped.nunique()), em_ext_reactions=int(EM.rxnsmiles_ordered.nunique()),
                   enzymes=int(P.Entry.nunique()), em_ext_enzymes=int(EM.Entry.nunique()), exact=len(P) == len(EM))

    # ---- R2
    old = json.load(open(a.old_vocab)); old_by_id = collections.defaultdict(set)
    for t in old:
        if not t.startswith('<') and t not in ('>>', '.') and ident(t) is not None: old_by_id[ident(t)].add(t)
    def snap(t, keep_single_atom_ions=False):
        if t == '[H+]' or ident(t) is None: return t
        if keep_single_atom_ions and Chem.MolFromSmiles(t).GetNumHeavyAtoms() == 1: return t
        c = old_by_id.get(ident(t))
        if not c or t in c: return t
        return next(iter(c)) if len(c) == 1 else min(c, key=lambda x: old[x])
    ex = set(EM.rxnsmiles_ordered); R['R2'] = {}
    for label, flag in (('plain_rule', False), ('REJECTED_blanket_single_atom_ion_exception', True)):
        pred = {u: '.'.join(snap(t, flag) for t in sides(u)[0]) + '>>' + '.'.join(snap(t, flag) for t in sides(u)[1]) for u in P.unmapped.unique()}
        hit = sum(1 for p in pred.values() if p in ex)
        R['R2'][label] = dict(reactions=len(pred), reproduced_exactly=hit, pct=round(100 * hit / len(pred), 2), em_ext_strings_not_produced=len(ex - set(pred.values())))

    # ---- R3
    data_toks = {t for s in ex for sd in sides(s) for t in sd}
    new = json.load(open('%s/vocab_em_ext.json' % a.em_ext_dir)); rebuilt = dict(old)
    for t in sorted(data_toks - set(old)): rebuilt[t] = len(rebuilt)
    R['R3'] = dict(rebuilt_equals_shipped=rebuilt == new, old_size=len(old), new_size=len(new), appended=len(new) - len(old))

    # ---- R4 (sizes only)
    N = len(EM); sz = EM.groupby('split').size().to_dict()
    R['R4'] = dict(rows=sz, fractions={k: round(100 * v / N, 3) for k, v in sz.items()}, train_target=round(0.8 * N, 1), train_cum_target_ok=sz['train'] <= 0.8 * N,
                   trainval_target=round(0.9 * N, 1), trainval_ok=sz['train'] + sz['val'] <= 0.9 * N,
                   cross_split_reaction_overlap=int(len(set(tr.rxnsmiles_ordered) & set(te.rxnsmiles_ordered)) + len(set(tr.rxnsmiles_ordered) & set(va.rxnsmiles_ordered)) + len(set(va.rxnsmiles_ordered) & set(te.rxnsmiles_ordered))),
                   seed_recovered=False)
    json.dump(R, open(a.out_json, 'w'), indent=1); print(json.dumps(R, indent=1))

if __name__ == '__main__':
    main()
