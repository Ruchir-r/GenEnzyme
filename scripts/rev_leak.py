import csv, sys, json
sys.path.insert(0, '/path/to/struct_rxnpred/claude_dir/17_reaction_revival')
from full_rxn_audit import side
E = '/path/to/struct_rxnpred/data/em_ext/em_ext_'
def load(s): return list(csv.DictReader(open(E + s + '.clean.csv')))
def key(rx):
    p = rx.split('>>'); a, b = side(p[0]), side(p[-1])
    return None if a is None or b is None else (a[0], b[0])
tr = load('train'); va = load('val'); te = load('test')
ktr = set(k for k in (key(r['rxnsmiles_ordered']) for r in tr) if k)
kva = set(k for k in (key(r['rxnsmiles_ordered']) for r in va) if k)
out = [['row_idx','rev_in_train','rev_in_val','fwd_in_val']]
n1 = n2 = n3 = 0
for i, r in enumerate(te):
    k = key(r['rxnsmiles_ordered'])
    rv = (k[1], k[0])
    a = int(rv in ktr); b = int(rv in kva); f = int(k in kva)
    n1 += a; n2 += b; n3 += f
    out.append([i, a, b, f])
w = csv.writer(open('/path/to/struct_rxnpred/claude_dir/17_reaction_revival/audit/rev_leak.csv', 'w', newline='')); w.writerows(out)
print('REVLEAK rows', len(te), 'reverse in train', n1, 'reverse in val', n2, 'forward in val', n3, flush=True)
