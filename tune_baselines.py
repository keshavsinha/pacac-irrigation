"""Tune the threshold and MPC references on development scenarios (seeds 5000-5009),
disjoint from training (>=1e6) and held-out test scenarios (1-20).
Selection rule (same as for learned controllers' reported operating point): maximise mean
nominal return subject to mean water <= W_budget, grid energy <= E_budget, soil <= d_soil."""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim import Params, Spec, make_scenario, rollout
from controllers import Threshold, MPC

DEV = list(range(5000, 5010))


def evaluate(ctrl, seeds, p=Params(), spec=Spec()):
    ms = [rollout(ctrl, make_scenario(s, p), p, spec) for s in seeds]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def feasible(m, spec=Spec()):
    return m['water'] <= spec.W_budget and m['egrid'] <= spec.E_budget and m['csoil'] <= spec.d_soil


if __name__ == '__main__':
    what = sys.argv[1]
    os.makedirs('results', exist_ok=True)
    rows = []
    if what == 'threshold':
        for lo, gap, rate, day in itertools.product([0.19, 0.20, 0.21, 0.22, 0.23], [0.03, 0.05, 0.08, 0.10],
                                                    [0.3, 0.6, 0.9], [False, True]):
            m = evaluate(Threshold(lo, lo + gap, rate, day), DEV)
            rows.append(dict(lo=lo, hi=lo + gap, rate=rate, day=day, **m, feas=feasible(m)))
        feas = [r for r in rows if r['feas']]
        best = max(feas, key=lambda r: r['ret']) if feas else min(rows, key=lambda r: r['water'])
        json.dump(dict(rows=rows, best=best), open('results/tune_threshold.json', 'w'), indent=1)
        print('best', {k: best[k] for k in ['lo', 'hi', 'rate', 'day', 'ret', 'water', 'egrid', 'csoil', 'P_mean']})
    elif what == 'mpc':
        for lw, le, tw in itertools.product([0.5, 1.0, 2.0, 4.0], [0.5], [0.5, 2.0]):
            t0 = time.time()
            m = evaluate(MPC(lam_w=lw, lam_e=le, term_w=tw), DEV[:5])
            rows.append(dict(lam_w=lw, lam_e=le, term_w=tw, **m, feas=feasible(m)))
            print(lw, le, tw, round(m['ret'], 1), round(m['water'], 1), round(m['egrid'], 1), round(m['csoil'], 3),
                  f'{time.time()-t0:.0f}s', flush=True)
        feas = [r for r in rows if r['feas']]
        best = max(feas, key=lambda r: r['ret']) if feas else min(rows, key=lambda r: r['water'])
        json.dump(dict(rows=rows, best=best), open('results/tune_mpc.json', 'w'), indent=1)
        print('best', best)
