#!/bin/bash
cd /home/claude/pacac
ROB=nominal,noisy_kf,noisy_raw,bias+0.02,bias-0.02,dropout30,pest_misspec+3C,couple0,couple0.5,couple2,soil_Zr300,soil_clayloam,rain_x0.5,rain_x1.5,hot+2C,Topt+3C_true
python3 evaluate.py td3 $ROB &
( for m in pacac_nopstate pacac_noprew pacac_nopest pacac_noaug pacac_cgrad pacac_penalty; do python3 evaluate.py $m nominal,couple0; done
  for m in sens_a4x2 sens_a4x0.5 sens_Wx0.85 sens_Wx1.15; do python3 evaluate.py $m nominal; done
  python3 evaluate.py td3lag nominal,couple0 ) &
wait
python3 eval_layered.py & python3 deploy.py & wait
python3 make_results.py
