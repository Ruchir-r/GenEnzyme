#!/bin/bash
#SBATCH --job-name=hzeval
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/struct_rxnpred/claude_dir/18_horizyn/results/slurm_hzeval_%j.out
#SBATCH --error=/path/to/struct_rxnpred/claude_dir/18_horizyn/results/slurm_hzeval_%j.err

R2=/path/to/struct_rxnpred
source $R2/claude_dir/17_reaction_revival/env_cache.sh || true

D=$R2/claude_dir/18_horizyn
H=$D/horizyn
py=$HOME/.conda/envs/torch-gpu-copy/bin/python3
cd $H

# Checkpoints are written by train_emext.py to $D/checkpoints/emext (NOT $H/...).
# A previous run of this script probed the wrong directory, skipped both arms and
# still exited 0 -- so a missing checkpoint is now recorded and made fatal at the
# end, while any checkpoint that IS present is still evaluated.
MISSING=0

for NAME in hz_emext_bs16384_e100 hz_emext_bs4096_e300; do
  CKPT=$D/checkpoints/emext/$NAME.ckpt
  if [ ! -f "$CKPT" ]; then
    echo "MISSING checkpoint $CKPT -- skipping"; MISSING=1; continue
  fi

  echo "=== [$NAME] caching reaction embeddings for the expanded pool ==="
  $py scripts/cache_reactions.py \
      --reactions data/em_ext_expanded_rxnpool.csv \
      --out data/rxncache_$NAME.h5 \
      --checkpoint $CKPT \
      --config configs/sota.yaml \
      --device cpu || exit 1

  echo "=== [$NAME] reversed screening: enzyme query -> ranked reaction pool ==="
  $py scripts/screen_reactions.py \
      --protein-h5 data/em_ext_testprots.h5 \
      --reaction-cache data/rxncache_$NAME.h5 \
      --checkpoint $CKPT \
      --config configs/sota.yaml \
      --top-k 200 \
      --device cpu \
      --output results_screen_$NAME.csv || exit 1

  echo "=== [$NAME] scoring (same strata as the pretrained-checkpoint arm) ==="
  $py $D/score_screen_v2.py \
      results_screen_$NAME.csv \
      $R2/data/em_ext/em_ext_test.clean.csv \
      $D/pair_leakage_leaked_pairs.json \
      $D/pair_leakage_rxnseen_pairs.json \
      $D/dual_clean_pairs.json \
      $D/score_$NAME.json || exit 1
done

if [ "$MISSING" -ne 0 ]; then
  echo "=== FAILED: at least one checkpoint was absent ==="; exit 1
fi
echo "=== DONE ==="
