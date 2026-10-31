# Data sources and licences

| source | used for | licence | where |
|---|---|---|---|
| EnzymeMap v2 (BRENDA 2023 release) | reactions and EC annotations behind EnzymeMap_ext | CC BY 4.0 (dataset record); code MIT | https://zenodo.org/records/8254726 , https://github.com/hesther/enzymemap |
| Rhea | reaction and charge-form conventions | CC BY 4.0 | https://www.rhea-db.org/help/license-disclaimer |
| UniProt / Swiss-Prot | accessions, protein sequences | CC BY 4.0 | https://www.uniprot.org/help/license |
| AlphaFold Protein Structure Database | predicted structures used as model input (not redistributed here) | CC BY 4.0 | https://alphafold.ebi.ac.uk/ |

Cite the EnzymeMap paper for the reaction data: Heid, Probst, Green, Madsen, *EnzymeMap: curation, validation and
data-driven prediction of enzymatic reactions*, Chem. Sci. 14, 14229-14242 (2023), doi:10.1039/d3sc02048g.

## Access dates
Rhea / UniProt / Swiss-Prot snapshot: 15 September 2025, as recorded in the paper's Methods §1.1 (Rhea + UniProt/SwissProt corpus).
This date is taken from the author's notes and was not independently checked when this page was written. The
retrieval date of the UniProt sequences joined to EnzymeMap_ext is not recorded here.

## What is redistributed
`data/enzymemap_ext/` is derived from EnzymeMap (reactions) joined to UniProt accessions (sequences) and is
released under the same CC BY 4.0 attribution terms as its sources. AlphaFold structures and ESM-2 embeddings are
not included; they are fetched or computed from the accessions.

## Not checked here
BRENDA's own terms (EnzymeMap is built from BRENDA; the CC BY 4.0 licence on the EnzymeMap record is what
permits redistribution of that dataset) and the licence of the ESM-2 weights.
