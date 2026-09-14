#!/bin/bash
# Long-budget finalist runs (candidate final configs).
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne/scripts/experiments
L() { echo "=== $(date +%H:%M:%S) $* ==="; }

L L_mlp_long
python emulator_lab.py --name L_mlp_long --arch mlp --depth 5 --width 512 \
  --mass-norm --sfr-arsinh --fourier-k 16 --batch 2048 --epochs 2400 --save-weights
L L_res_long
python emulator_lab.py --name L_res_long --arch resmlp --blocks 4 --width 512 \
  --mass-norm --sfr-arsinh --fourier-k 16 --batch 2048 --epochs 2400 --save-weights
L DONE
