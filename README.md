# GenEnzyme

Generative prediction of enzymatic reactions from the enzyme alone, with an exact, reaction-disjoint evaluation.

> **Status: under construction.** This repository is being prepared alongside the accompanying manuscript. Code, data splits and documentation are being added and may change without notice. Nothing here is final, and results should not be reproduced or cited from this repository until this notice is removed.

## What this project is

GenEnzyme accompanies the manuscript *Generative Models for Novel Enzymatic Reactions: Recombination and Its Limits*. It studies a structure-conditioned generative model that writes a complete reaction (reactants and products) from an enzyme's sequence and structure, and evaluates it under exact, stereo-respecting scoring on a reaction-disjoint benchmark, EnzymeMap_ext, in which no test reaction appears in training.

## Planned contents

The following are intended for release with the paper:

- **Benchmark splits** for EnzymeMap_ext (train / validation / test, reaction-disjoint).
- **Difficulty-class assignment** (`exact_seen`, `recombination`, `novel_metabolite`, with the `novel_product` and `novel_reactant_only` refinements).
- **Audit scorer** (`full_rxn_audit.py`): stereo-respecting multiset comparison, product-set and full-reaction recovery, Wilson and enzyme-clustered bootstrap intervals.
- **Free-running re-ranking** (assumption-free and novelty-assuming variants).
- **Comparison scripts** for the homology baselines (BLASTp, DIAMOND) and the retrained RxnPred and Horizyn comparisons.
- **Model code** for the sequence-and-structure encoder and the reaction decoder.

## Repository layout

To be added.

## Usage

To be added.

## Data and licensing

EnzymeMap_ext is derived from the public EnzymeMap database. Data sources, licences and redistribution terms will be documented here before release.

## Citation

A preprint reference will be added when available.

## Authors

Ruchir R, Srikrishnan Balamukundhan and Karthik Raman, Department of Data Science and Artificial Intelligence, IIT Madras, Chennai, India.
