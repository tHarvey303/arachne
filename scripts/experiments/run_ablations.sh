#!/bin/bash
# Ablation series for ParrotEmulator improvements. Run on a GPU node.
source /cosma/apps/dp276/dc-harv3/venv_simformer/bin/activate
cd /cosma/apps/dp276/dc-harv3/arachne/scripts/experiments

L() { echo "=== $(date +%H:%M:%S) $* ==="; }

L ref_checkpoint
python emulator_lab.py --name ref_checkpoint --eval-checkpoint /cosma/apps/dp276/dc-harv3/arachne/scripts/outputs/emulators/parrot_emulator.eqx
L A0_base
python emulator_lab.py --name A0_base --arch mlp --depth 5 --width 512
L A1_mass
python emulator_lab.py --name A1_mass --arch mlp --depth 5 --width 512 --mass-norm
L A2_mass_sfr
python emulator_lab.py --name A2_mass_sfr --arch mlp --depth 5 --width 512 --mass-norm --sfr-arsinh
L A3_mass_sfr_f16
python emulator_lab.py --name A3_mass_sfr_f16 --arch mlp --depth 5 --width 512 --mass-norm --sfr-arsinh --fourier-k 16
L A4_res
python emulator_lab.py --name A4_res --arch resmlp --blocks 4 --width 512 --mass-norm --sfr-arsinh --fourier-k 16
L A5_res_nofourier
python emulator_lab.py --name A5_res_nofourier --arch resmlp --blocks 4 --width 512 --mass-norm --sfr-arsinh
L A6_ema
python emulator_lab.py --name A6_ema --arch mlp --depth 5 --width 512 --mass-norm --sfr-arsinh --fourier-k 16 --ema 0.999
L DONE
