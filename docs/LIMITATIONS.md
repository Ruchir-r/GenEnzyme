# Limitations

1. **Not enzyme-disjoint.** 818 of the 1,277 test enzymes (861 counting val) also appear in train. Held-out
   reactions are new for the model; held-out enzymes mostly are not.
2. **Exact-string disjointness only.** Against the training split, 30 of the 1,072 distinct test reactions
   (41 of 2,349 test rows) equal a training reaction once stereochemistry is dropped, and 24 distinct reactions
   (33 rows) are the exact reverse of a training reaction under the strict stereo key. Recomputed with RDKit
   2026.09.1 while assembling this release.
3. **Transductive vocabulary.** The vocabulary was built over the whole corpus. 749 tokens occur in test but never in train;
   those tokens have ids but receive no training signal. The `novel_metabolite` class is where this matters.
4. **Mixed charge forms.** Molecules found in an older vocabulary use its charge form; others keep EnzymeMap's
   neutral string. About 60% of reactions are charge-balanced, so charge-conservation checks do not apply.
   Heavy-atom balance (hydrogen excluded) is unaffected.
5. **Single seed.** Every model arm was trained once (seed 0). Arm differences, including the
   structure-versus-sequence ablation, are not replicated across seeds.
6. **The lookup arm is not deployable.** The re-ranker that removes candidates equal to a known train/val
   reaction reaches 5.62% full-reaction top-1 (132 of 2,349) only because test reactions are guaranteed absent from
   train. Without that lookup, top-1 is 5 of 2,349 (0.21%); see `results/tables/rerank_by_budget.csv`.
7. **Cluster-specific code.** Scripts are released as run, with absolute cluster paths replaced by placeholders (`/path/to/struct_rxnpred`). They show what produced
   the numbers; they are not yet path-portable. See REPRODUCE.md.
