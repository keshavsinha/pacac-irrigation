"""Simulator verification: (i) water-balance closure, (ii) action-response tests,
(iii) summary of exogenous weather statistics over the 20 test scenarios."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim import Params, Spec, make_scenario, init_state, step, N_STEPS, STEPS_PER_DAY

p = Params()
out = {}
# (i) closure over a rollout with a fixed irrigation pattern
errs = []
for seed in range(1, 21):
    sc = make_scenario(seed, p); s = init_state(sc)
    inflow = outflow = 0.0
    S0 = s['th'] * p.Zr
    for t in range(N_STEPS):
        u = 0.45 if (sc['hour'][t] > 9 and sc['hour'][t] < 12) else 0.0
        s, info = step(s, u, t, sc, p)
        inflow += u - info['loss'] + info['rain']
        outflow += info['etc'] + info['D'] + info['runoff']
    errs.append(abs((s['th'] * p.Zr - S0) - (inflow - outflow)))
out['closure_max_abs_mm'] = float(max(errs))

# (ii) action response: single 4.5 mm pulse (5 steps at u_max) starting 02:00 vs 11:00 on day 3
def pulse(start_hour, seed=3):
    sc = make_scenario(seed, p)
    res = {}
    for amt in [0, 1]:
        s = init_state(sc); Ps, ths, Hs, ws = [], [], [], []
        t0 = 2 * STEPS_PER_DAY + int(start_hour * 2)
        for t in range(N_STEPS):
            u = p.u_max * amt if t0 <= t < t0 + 5 else 0.0
            s, info = step(s, u, t, sc, p)
            Ps.append(s['P']); ths.append(s['th']); Hs.append(s['H']); ws.append(s['w'])
        res[amt] = dict(P=np.array(Ps), th=np.array(ths), H=np.array(Hs), w=np.array(ws))
    win = slice(t0, t0 + 48)
    return dict(dtheta=float(res[1]['th'][t0 + 6] - res[0]['th'][t0 + 6]),
                dH_peak=float((res[1]['H'] - res[0]['H'])[win].max()),
                wet_hours=float(((res[1]['w'] > 0.5)[win].sum() - (res[0]['w'] > 0.5)[win].sum()) * 0.5),
                dP_24h=float((res[1]['P'] - res[0]['P'])[t0 + 47]))
out['pulse_night_0200'] = pulse(2.0)
out['pulse_day_1100'] = pulse(11.0)
# (iii) weather summary
W = dict(rain=[], wet_days=[], Tmin=[], Tmax=[], RHmin=[], RHmax=[], et0=[])
for seed in range(1, 21):
    sc = make_scenario(seed, p)
    W['rain'].append(sc['rain_total']); W['wet_days'].append(sc['wet_days'])
    W['Tmin'].append(sc['T'][:N_STEPS].min()); W['Tmax'].append(sc['T'][:N_STEPS].max())
    W['RHmin'].append(sc['RH'][:N_STEPS].min()); W['RHmax'].append(sc['RH'][:N_STEPS].max())
    W['et0'].append(sc['et0'][:N_STEPS].sum())
out['weather'] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in W.items()}
json.dump(out, open('results/validation.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
