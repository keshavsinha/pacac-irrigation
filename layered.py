"""Structural-mismatch test: 4-layer (4 x 100 mm) vertical cascade soil model with
root-uptake distribution 40/30/20/10 %, layer-wise drainage and a single VWC probe at a
chosen layer. Controllers trained on the lumped model observe only the probe reading;
agronomic metrics use the root-weighted mean VWC. (Intermediate check, not HYDRUS.)"""
import numpy as np
from sim import (Params, Spec, N_STEPS, init_state, step, reward, supervisor, ks_coef, Estimator, Sensing)

ROOT = np.array([0.4, 0.3, 0.2, 0.1])


def rollout_layered(controller, sc, p=Params(), spec=Spec(), probe=1, kd_layer=0.08):
    s = init_state(sc)
    L = np.full(4, sc['th0'])
    dz = p.Zr / 4
    controller.reset(sc, p, spec)
    est_m = Estimator(sc, p, Sensing())
    tot = dict(ry=0.0, err=0.0, Ks=0.0, sup=0, viol=0, Psum=0.0, csoil=0.0, wsum=0.0, egrid=0.0,
               top_sat=0, low_def=0)
    for t in range(N_STEPS):
        th_mean = float(ROOT @ L)
        s['th'] = th_mean
        s_obs = dict(s); s_obs['th'] = L[probe]
        est = est_m.update(t, s_obs, None, None)
        u_cmd = controller.act(est, s_obs, t, sc)
        u, code = supervisor(u_cmd, L[probe], p)
        u = float(u)
        s1, info = step(s, u, t, sc, p)          # microclimate, disease, NDVI, energy
        # layered water redistribution (replaces lumped soil update)
        inflow = info['u'] - info['loss'] + info['rain']
        Ks_l = ks_coef(L, p)
        et_l = ROOT * Ks_l * p.Kc * sc['et0'][t]
        L = L - et_l / dz
        for i in range(4):
            L[i] += inflow / dz
            exc = max(L[i] - p.th_fc, 0) * kd_layer * dz
            sat = max(L[i] - p.th_s, 0) * dz
            L[i] -= (exc + sat) / dz
            inflow = exc + sat
        th1 = float(ROOT @ L)
        s1['th'] = th1
        c_soil = max(th1 - p.th_max, 0) + max(p.th_min - th1, 0)
        s1['Csoil'] = s['Csoil'] + c_soil
        r, ry = reward(s1, info, p, spec)
        tot['ry'] += ry; tot['err'] += abs(th1 - p.th_star); tot['Ks'] += float(ROOT @ Ks_l)
        tot['sup'] += code > 0; tot['viol'] += c_soil > 0; tot['Psum'] += s1['P']
        tot['top_sat'] += L[0] > p.th_max; tot['low_def'] += L[3] < p.th_min
        controller.observe(u, info)
        s = s1
    return dict(water=s['W'], egrid=s['E'], csoil=s['Csoil'], mae_theta=tot['err'] / N_STEPS,
                rel_transp=tot['Ks'] / N_STEPS, P_mean=tot['Psum'] / N_STEPS, sup=tot['sup'],
                viol_steps=tot['viol'], top_sat_steps=tot['top_sat'], deep_def_steps=tot['low_def'])
