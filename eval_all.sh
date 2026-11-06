#!/bin/bash
# Evaluate every controller on the 20 held-out scenarios (nominal + robustness conditions)
cd /home/claude/pacac
ROB=nominal,noisy_kf,noisy_raw,bias+0.02,bias-0.02,dropout30,pest_misspec+3C,couple0,couple0.5,couple2,soil_Zr300,soil_clayloam,rain_x0.5,rain_x1.5,hot+2C,Topt+3C_true
python3 evaluate.py thr_classic $ROB &
python3 evaluate.py thr_tuned $ROB &
wait
python3 evaluate.py pacac $ROB &
python3 evaluate.py td3 $ROB &
wait
for m in td3lag pacac_noaug pacac_nopstate pacac_noprew pacac_nopest; do
  python3 evaluate.py $m nominal,couple0 &
done
wait
python3 evaluate.py mpc nominal,couple0,noisy_kf,rain_x0.5,rain_x1.5
