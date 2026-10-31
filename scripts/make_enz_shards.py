import csv
R2='/path/to/struct_rxnpred'; D17='/path/to/struct_rxnpred/claude_dir/17_reaction_revival'
def load(s): return list(csv.DictReader(open(R2+'/data/em_ext/em_ext_'+s+'.clean.csv')))
tr, va, te = load('train'), load('val'), load('test')
test_ent = set(r['Entry'] for r in te)
first = {}
for r in tr + va:
    if r['Entry'] not in test_ent and r['Entry'] not in first: first[r['Entry']] = r
ents = sorted(first)
hdr = list(tr[0].keys())
for tag, part in (('A', ents[0::2]), ('B', ents[1::2])):
    w = csv.DictWriter(open(D17+'/final_eval/enz_pool_'+tag+'.clean.csv','w',newline=''), fieldnames=hdr); w.writeheader()
    for e in part: w.writerow(first[e])
    print('shard', tag, len(part))
print('non-test enzymes', len(ents))
