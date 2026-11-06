"""Held-out evaluation: every controller on the same 20 pre-generated test scenarios (seeds 1-20).
Writes one CSV row per (method, training seed, scenario, condition)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from dataclasses import replace
from sim import Params, Spec, Sensing, make_scenario, rollout
from controllers import Threshold, MPC, Policy

TEST = list(range(1, 21))


class LoadedActor:
    def __init__(self, path):
        z = np.load(path)
        arrs = [z[f'arr_{i}'] for i in range(len(z.files) - 1)]
        self.W = [(arrs[i], arrs[i + 1]) for i in range(0, len(arrs), 2)]
        self.u_max = float(z['u_max'])

    def __call__(self, o):
        x = o
        for k, (W, b) in enumerate(self.W):
            x = W @ x + b
            if k < len(self.W) - 1:
                x = np.maximum(x, 0)
        return (np.tanh(x[0]) + 1) / 2 * self.u_max


def learned(name, seed):
    pre = f'results/models/{name}_s{seed}'
    h = json.load(open(pre + '_hist.json'))
    c = h['cfg']
    spec_obs = Spec(W_budget=c['W_budget'], E_budget=c['E_budget'], d_soil=c['d_soil'])
    return Policy(LoadedActor(pre + '_actor.npz'), h['keys'], spec_obs, c.get('repeat', 1)), c


def baseline(name):
    if name == 'thr_classic':
        return Threshold(0.20, 0.30, 0.9, False)
    if name == 'thr_tuned':
        b = json.load(open('results/tune_threshold.json'))['best']
        return Threshold(b['lo'], b['hi'], b['rate'], b['day'])
    if name == 'mpc':
        b = json.load(open('results/tune_mpc.json'))['best']
        return MPC(lam_w=b['lam_w'], lam_e=b['lam_e'], term_w=b['term_w'])
    raise KeyError(name)


CONDITIONS = {
    'nominal': dict(),
    'noisy_kf': dict(sen=dict(noisy=True)),
    'noisy_raw': dict(sen=dict(noisy=True, use_kf=False)),
    'bias+0.02': dict(sen=dict(noisy=True, bias_th=0.02)),
    'bias-0.02': dict(sen=dict(noisy=True, bias_th=-0.02)),
    'dropout30': dict(sen=dict(noisy=True, p_drop=0.3)),
    'pest_misspec+3C': dict(sen=dict(pest_misspec=3.0)),
    'couple0': dict(couple=0.0),
    'couple0.5': dict(couple=0.5),
    'couple2': dict(couple=2.0),
    'soil_Zr300': dict(par=dict(Zr=300.0)),
    'soil_clayloam': dict(par=dict(th_fc=0.34, th_wp=0.17, th_s=0.46, k_d=0.02)),
    'rain_x0.5': dict(par=dict(rain_mult=0.5)),
    'rain_x1.5': dict(par=dict(rain_mult=1.5)),
    'hot+2C': dict(par=dict(T_shift=2.0)),
    'Topt+3C_true': dict(par=dict(Topt_p=27.0)),
}


def run(method, ctrl, cond, tag_seed, extra=None):
    cd = CONDITIONS[cond]
    p = replace(Params(), **cd.get('par', {}))
    sen = Sensing(**cd.get('sen', {}))
    rows = []
    for s in TEST:
        sc = make_scenario(s, p)
        m = rollout(ctrl, sc, p, Spec(), sen, couple=cd.get('couple', 1.0))
        rows.append(dict(method=method, train_seed=tag_seed, scenario=s, cond=cond, **m, **(extra or {})))
    return rows


if __name__ == '__main__':
    method = sys.argv[1]
    conds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['nominal']
    out = f'results/eval/{method}.csv'
    os.makedirs('results/eval', exist_ok=True)
    rows = []
    if method in ('thr_classic', 'thr_tuned', 'mpc'):
        for cond in conds:
            rows += run(method, baseline(method), cond, 0)
    else:
        for f in sorted(glob.glob(f'results/models/{method}_s*_actor.npz')):
            seed = int(f.split('_s')[-1].split('_')[0])
            ctrl, _ = learned(method, seed)
            for cond in conds:
                rows += run(method, ctrl, cond, seed)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out, len(rows))
