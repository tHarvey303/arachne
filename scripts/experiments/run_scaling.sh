#!/bin/bash
# Data-scaling study: winner arch + original arch at 100k/300k/925k train rows.
# Epochs scaled inversely with n_train => constant optimisation budget (~180k steps).
# Usage: bash run_scaling.sh "<winner flags>"
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne/scripts/experiments
WINNER="$1"

L() { echo "=== $(date +%H:%M:%S) $* ==="; }

L S_win_100k
python emulator_lab.py --name S_win_100k $WINNER --n-train 100000 --epochs 7500
L S_win_300k
python emulator_lab.py --name S_win_300k $WINNER --n-train 300000 --epochs 2400
# winner at 925k = the ablation/HPO run itself (reuse)

L S_base_100k
python emulator_lab.py --name S_base_100k --arch mlp --depth 5 --width 512 --n-train 100000 --epochs 7500
L S_base_300k
python emulator_lab.py --name S_base_300k --arch mlp --depth 5 --width 512 --n-train 300000 --epochs 2400

# Seed ensemble of the winner at full data (for noise-floor analysis)
L S_win_seed1
python emulator_lab.py --name S_win_seed1 $WINNER --seed 1
L S_win_seed2
python emulator_lab.py --name S_win_seed2 $WINNER --seed 2
L DONE
