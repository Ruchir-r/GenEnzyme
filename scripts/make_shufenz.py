import csv, random, collections
R2='/path/to/struct_rxnpred'
src=R2+'/data/em_ext/em_ext_test.clean.csv'
dst=R2+'/claude_dir/17_reaction_revival/final_eval/em_ext_test_shufenz.clean.csv'
rows=list(csv.DictReader(open(src)))
ents=[r['Entry'] for r in rows]
rnd=random.Random(0)
perm=ents[:]
rnd.shuffle(perm)
n=len(ents)
for it in range(100000):
    bad=[i for i in range(n) if perm[i]==ents[i]]
    if not bad: break
    i=bad[0]
    while True:
        j=rnd.randrange(n)
        if perm[j]!=ents[i] and perm[i]!=ents[j]:
            perm[i],perm[j]=perm[j],perm[i]; break
assert all(perm[i]!=ents[i] for i in range(n))
assert collections.Counter(perm)==collections.Counter(ents)
for r,e in zip(rows,perm): r['Entry']=e
w=csv.DictWriter(open(dst,'w',newline=''),fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(rows)
print('wrote',dst,'rows',n,'own-enzyme rows',sum(perm[i]==ents[i] for i in range(n)),'iters',it)
