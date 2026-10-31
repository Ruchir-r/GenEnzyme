# EnzymeMap_ext benchmark

## Contents
`data/enzymemap_ext/`

| file | description |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | columns `Entry` (UniProt accession), `rxnsmiles_ordered` (`reactants>>products`, metabolites joined by `.`) |
| `difficulty_classes.csv` | one class per **test** row: `row_idx`, `Entry`, `difficulty_class` |
| `metabolite_vocab.csv` | `token_id`, `token`; ids 0-6 are special tokens (`<START>`, `<END>`, `<PAD>`, `<MASK>`, `<UNK>`, `>>`, `.`), then one id per whole-metabolite SMILES |
| `enzyme_sequences.csv` | protein sequences for the accessions |

## Size
| split | rows | distinct enzymes | distinct reactions |
|---|---:|---:|---:|
| train | 18,758 | 5,752 | 8,940 |
| val | 2,344 | 1,357 | 1,076 |
| test | 2,349 | 1,277 | 1,072 |
| union | 23,451 | 6,649 | 11,088 |

Rows are (enzyme, reaction) pairs. The vocabulary has 16053 entries (7 special + metabolite tokens).

## Splits
Reactions are **disjoint by exact string** across train, val and test (0 shared in every pair). The split is
**not** disjoint by enzyme (see LIMITATIONS.md) and is not a similarity-clustered split.
The script that assigned rows to splits was not recovered; `scripts/reconstruct_em_ext.py` rebuilds the corpus,
vocabulary and charge-form rules from public EnzymeMap but does **not** reproduce the split assignment.
The released split files are the authoritative definition.

## Difficulty classes (test set)
Computed by `scripts/assign_difficulty.py` against the training split. Keys are multisets of canonical SMILES.

| class | rule | test rows |
|---|---|---:|
| `exact_seen` | the reaction key occurs in train | 0 |
| `recombination` | key unseen; every reactant and product metabolite occurs somewhere in train | 1,275 |
| `novel_metabolite` | at least one metabolite never occurs in train | 1,074 |

`exact_seen` is empty on test by construction (reaction-disjoint split).

## Scoring contract
A candidate pool has columns `row_idx`, `rank` (or file order within a row), `True SMILES`, `Pred SMILES`.
`scripts/full_rxn_audit.py` records, per test row, the first rank at which each criterion holds, so recall@K is
`first_rank >= 0 and first_rank < K`. Components are compared as **sorted multisets of RDKit-canonical SMILES**.

| criterion | meaning |
|---|---|
| `full_strict` | exactly one `>>`; reactant multiset AND product multiset equal the truth, stereochemistry respected (isomeric SMILES) |
| `prod_strict` | product multiset equal, stereochemistry respected |
| `full_const`, `prod_const` | same, stereochemistry removed (a looser, secondary metric) |
| `react_strict` | generated reactant multiset equals the true reactant multiset |

The headline metric is `full_strict`. `tests/test_audit_scorer.py` checks the comparison on enantiomers,
ordering, stoichiometry and unparseable input.
