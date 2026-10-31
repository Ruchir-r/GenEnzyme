# GenEnzyme

Code, benchmark and scorer accompanying a study of structure-conditioned generative prediction of enzyme
reaction products (enzyme in, reaction out). The repository releases the **EnzymeMap_ext** splits and
difficulty classes, the stereo-aware scorer used for every recovery number, the code as it was run, and the
tables behind the re-ranking results.

On the reaction-disjoint EnzymeMap_ext benchmark, the model recovers the complete reaction for 14.3% of test
enzyme-reaction pairs within 400 samples (25.0% for recombinations of metabolites seen in training, 1.6% for reactions
that need an unseen metabolite). A matched ablation finds no measurable effect of structure on exact recovery.

## Contents
| path | what |
|---|---|
| `data/enzymemap_ext/` | train / val / test splits, `difficulty_classes.csv`, `metabolite_vocab.csv`, enzyme sequences |
| `scripts/full_rxn_audit.py` | candidate-pool scorer (strict full-reaction and product-set recovery) |
| `scripts/assign_difficulty.py` | assigns `recombination` / `novel_metabolite` / `exact_seen` classes |
| `scripts/` | re-ranking, bucketed scoring, baselines, sweep worker (as run on the cluster) |
| `src/genenzyme/` | model, data pipeline, training and inference code; `stats.py` (Wilson, McNemar, clustered bootstrap). The model files other than the deployed configuration are development variants; `models/mhggat_rag.py` is a Foldseek retrieval-augmented experiment that is not part of the deployed model |
| `configs/cbv2_t010.yaml` | hyperparameters of the deployed model |
| `results/tables/` | CSVs behind the re-ranking results and the architecture sweep |
| `docs/` | `REPRODUCE.md`, `BENCHMARK.md`, `DATA_SOURCES.md` |
| `tests/test_audit_scorer.py` | unit tests for the stereo-respecting multiset comparison |

## Quickstart
```bash
conda env create -f environment.yml && conda activate genenzyme
python scripts/assign_difficulty.py --train data/enzymemap_ext/train.csv --test data/enzymemap_ext/test.csv --out difficulty_classes.csv
python tests/test_audit_scorer.py
```
The second command should print `DIFFICULTY_COUNTS {'novel_metabolite': 1074, 'recombination': 1275} n = 2349`;
the third, `RESULT all passed`. See `docs/REPRODUCE.md` for what each paper number needs and what this
repository does not cover.

## Headline numbers shipped as tables
Free-running generation, 400 samples per test reaction (T = 1.0), n = 2,349 test rows, full reaction
(reactants and products, stereochemistry respected):

| | top-1 | top-10 |
|---|---:|---:|
| draw order | 2 (0.09%) | 53 (2.26%) |
| re-ranked, with train/val lookup | 132 (5.62%) | 291 (12.39%) |
| re-ranked, without lookup | 5 (0.21%) | not tabulated here |

The truth appears somewhere in the 400 draws for 336 of 2,349 rows (14.30%). The lookup variant removes candidates
equal to a known train or val reaction, which is valid on this benchmark only because test reactions are absent
from train by construction; the no-lookup row is the figure that applies to a deployed setting.
Sources: `results/tables/rerank_free_policies_summary.csv`, `results/tables/rerank_by_budget.csv`.

## Data and licence
Code: MIT (`LICENSE`). Data in `data/`: CC BY 4.0 (`data/LICENSE.md`); it is derived from CC BY 4.0 sources, see
`docs/DATA_SOURCES.md` for provenance and attribution.

## Citation
Cite the paper (preprint / DOI to be added once available) and, for the underlying reaction data, the EnzymeMap paper listed in
`docs/DATA_SOURCES.md`.
