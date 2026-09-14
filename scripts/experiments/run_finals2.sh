#!/bin/bash
# Second wave of long-budget finalists: fourier-64 at batch 2048.
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne/scripts/experiments
L() { echo "=== $(date +%H:%M:%S) $* ==="; }

L F_res768_f64
python emulator_lab.py --name F_res768_f64 --arch resmlp --blocks 6 --width 768 \
  --mass-norm --sfr-arsinh --fourier-k 64 --lr 0.00064 --batch 2048 --epochs 2400 --save-weights
L DONE
