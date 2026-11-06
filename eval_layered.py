"""Evaluate controllers (trained on the lumped model) on the 4-layer cascade soil model with the
VWC probe at each of the four depths (0-10, 10-20, 20-30, 30-40 cm)."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sim import Params, Spec, make_scenario
from layered import rollout_layered
from evaluate import learned, baseline, TEST

rows = []
methods = [('thr_classic', None), ('thr_tuned', None)]
for me in ['pacac', 'td3']:
    for f in sorted(glob.glob(f'results/models/{me}_s*_actor.npz')):
        methods.append((me, int(f.split('_s')[-1].split('_')[0])))
for me, seed in methods:
    ctrl = baseline(me) if seed is None else learned(me, seed)[0]
    for probe in range(4):
        for s in TEST:
            m = rollout_layered(ctrl, make_scenario(s), Params(), Spec(), probe=probe)
            rows.append(dict(method=me, train_seed=seed if seed is not None else 0, scenario=s, probe=probe, **m))
pd.DataFrame(rows).to_csv('results/eval_layered.csv', index=False)
print('ok', len(rows))
