#!/bin/bash
#SBATCH --job-name=homolbl
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/struct_rxnpred/claude_dir/18_horizyn/blast_baseline/slurm_homol_%j.out
#SBATCH --error=/path/to/struct_rxnpred/claude_dir/18_horizyn/blast_baseline/slurm_homol_%j.err
#
# Sequence-homology retrieval baseline (BLASTp + DIAMOND) for the em_ext
# product-recall comparison, structurally matched to Horizyn's reversed
# screening: test enzyme -> ranked train reactions -> product recall@k.
#
# CPU-only (small partition). Both searches query 1277 test enzymes against a
# 5751-enzyme train DB; permissive e-value and large max-target-seqs so the
# ranked hit list is deep enough for k up to 200.

set -euo pipefail
BT=/path/to/scratch/bio_tools
BP=$BT/ncbi-blast-2.17.0+/bin
DIA=$BT/diamond
W=/path/to/struct_rxnpred/claude_dir/18_horizyn/blast_baseline
VPY=/path/to/scratch/rxnmap_venv/bin/python
cd $W

echo "=== homology baseline job=${SLURM_JOB_ID} host=$(hostname) $(date -Is) ==="
NT=${SLURM_CPUS_PER_TASK:-16}

# ---- BLASTp -------------------------------------------------------------
if [ ! -s blastp_hits.tsv ]; then
  echo "--- makeblastdb $(date -Is)"
  $BP/makeblastdb -in train_db.fasta -dbtype prot -out train_blastdb >/dev/null
  echo "--- blastp $(date -Is)"
  $BP/blastp -query test_query.fasta -db train_blastdb \
    -outfmt 6 -evalue 10 -max_target_seqs 500 -num_threads $NT \
    -out blastp_hits.tsv
fi
echo "blastp rows: $(wc -l < blastp_hits.tsv)"

# ---- DIAMOND ------------------------------------------------------------
if [ ! -s diamond_hits.tsv ]; then
  echo "--- diamond makedb $(date -Is)"
  $DIA makedb --in train_db.fasta -d train_diadb --quiet
  echo "--- diamond blastp $(date -Is)"
  $DIA blastp -q test_query.fasta -d train_diadb \
    --outfmt 6 --evalue 10 --max-target-seqs 500 --very-sensitive \
    -p $NT -o diamond_hits.tsv --quiet
fi
echo "diamond rows: $(wc -l < diamond_hits.tsv)"

# ---- score both ---------------------------------------------------------
echo "--- score blastp $(date -Is)"
$VPY score_homology.py blastp_hits.tsv  train_pool.json test_truth.json score_blastp  BLASTp
echo "--- score diamond $(date -Is)"
$VPY score_homology.py diamond_hits.tsv train_pool.json test_truth.json score_diamond DIAMOND
echo "=== finished $(date -Is) ==="
