
import json, math
from itertools import product
hv = json.load(open("/path/to/struct_rxnpred/claude_dir/17_reaction_revival/hitvecs.json"))

def binom_two_sided(k, n):
    if n == 0: return 1.0
    # exact two-sided binomial test p=0.5
    def C(n,r):
        return math.comb(n,r)
    tot = 2.0**n
    pk = C(n,k)/tot
    s = 0.0
    for i in range(n+1):
        pi = C(n,i)/tot
        if pi <= pk + 1e-15: s += pi
    return min(1.0, s)

def mc(a,b,metric):
    A = hv[a][metric]; B = hv[b][metric]
    n01 = sum(1 for x,y in zip(A,B) if x==0 and y==1)
    n10 = sum(1 for x,y in zip(A,B) if x==1 and y==0)
    return sum(A), sum(B), n10, n01, binom_two_sided(n10, n10+n01)

pairs = [("chem_tf_beam5","fair_tf_beam5"),("chem_tf_beam5","seq24_tf_beam5"),
         ("fair_tf_beam5","seq24_tf_beam5"),("chem_tf_beam15","seq24_tf_beam15"),
         ("chem_tf_sample50","seq24_tf_sample50"),("chem_tf_sample10","fair_tf_sample10")]
for metric in ["t1","ak"]:
    for a,b in pairs:
        ha,hb,n10,n01,p = mc(a,b,metric)
        print("%-5s %-34s A=%4d B=%4d onlyA=%4d onlyB=%4d p=%.3g" % (metric, a+" vs "+b, ha,hb,n10,n01,p))
print()
for arm,bm,sm in [("chem","chem_tf_beam5","chem_tf_sample10"),
                  ("seq","seq24_tf_beam5","seq24_tf_sample10"),
                  ("fair","fair_tf_beam5","fair_tf_sample10")]:
    ha,hb,n10,n01,p = mc(bm,sm,"t1")
    print("BEAMvSAMP t1 %-6s beam=%4d samp=%4d onlyBeam=%4d onlySamp=%4d p=%.3g" % (arm,ha,hb,n10,n01,p))
