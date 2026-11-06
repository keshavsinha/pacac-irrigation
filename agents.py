"""Deterministic actor-critic learners with vectorised environments.

Methods (cost_mode):
  none      : TD3 (unconstrained, scalar reward)
  lag_disc  : TD3-Lag  - discounted cost critics (gamma_c=0.99) and dual update from minibatch
              averages of critic values over replay states (formulation of the original manuscript)
  pacac     : PAC-AC   - undiscounted finite-horizon cost critics on a time/budget-augmented state,
              dual update from Monte-Carlo episode constraint returns of the deterministic policy
  penalty   : fixed penalty kappa_i * c_i/d_i in the reward, no cost critics
  fixed_lam : PAC-AC cost critics with multipliers frozen at fixed_lam
All critics use n-step targets (n=24) that stop at the episode end.
"""
import numpy as np, torch, torch.nn as nn, copy, time, json
from sim import (Params, Spec, N_STEPS, make_scenario, step, reward, supervisor, build_obs,
                 OBS_ENV, OBS_AUG)

torch.set_num_threads(1)
HID = 64


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, o))


class Ensemble(nn.Module):
    """K independent 2x64 MLP critics evaluated with batched matmuls."""

    def __init__(self, K, inp):
        super().__init__()
        def init(i, o):
            w = torch.empty(K, i, o); b = torch.empty(K, 1, o)
            bound = 1 / np.sqrt(i)
            nn.init.uniform_(w, -bound, bound); nn.init.uniform_(b, -bound, bound)
            return nn.Parameter(w), nn.Parameter(b)
        self.w1, self.b1 = init(inp, HID)
        self.w2, self.b2 = init(HID, HID)
        self.w3, self.b3 = init(HID, 1)
        self.K = K

    def forward(self, x):                     # x: [B, inp] -> [K, B]
        h = torch.relu(torch.baddbmm(self.b1, x.unsqueeze(0).expand(self.K, -1, -1), self.w1))
        h = torch.relu(torch.baddbmm(self.b2, h, self.w2))
        return torch.baddbmm(self.b3, h, self.w3).squeeze(-1)


class NumpyActor:
    def __init__(self, net, u_max):
        self.W = [(l.weight.detach().numpy().copy(), l.bias.detach().numpy().copy())
                  for l in net if isinstance(l, nn.Linear)]
        self.u_max = u_max

    def __call__(self, o):
        x = o
        for k, (W, b) in enumerate(self.W):
            x = W @ x + b if x.ndim == 1 else x @ W.T + b
            if k < len(self.W) - 1:
                x = np.maximum(x, 0)
        a = np.tanh(x[..., 0])
        return (a + 1) / 2 * self.u_max

    def save(self, path):
        np.savez(path, *[a for wb in self.W for a in wb], u_max=self.u_max)


DEFAULT_CFG = dict(
    obs='env+aug', cost_mode='pacac', gamma=0.99, gamma_c=1.0, lr_a=3e-4, lr_c=3e-4, batch=256,
    tau=0.005, policy_delay=2, expl_sd=0.3, tgt_sd=0.2, tgt_clip=0.5, n_step=6, n_step_c=24, pre_reg=0.01,
    n_env=8, rounds=70, warmup_rounds=3, updates_per_step=4, repeat=4,
    lam_lr=3.0, lam_max=50.0, lam_window=2, n_dual=4, lam_lr_step=0.1,
    fixed_lam=None, penalty_kappa=None, lag_in_reward=True, no_pest_state=False, ckpt_every=2, n_dev=5,
    a1=1.0, a2=0.3, a3=0.3, a4=1.0, W_budget=38.0, E_budget=6.0, d_soil=0.15, seed=0, couple=1.0,
)


def obs_keys(cfg):
    keys = list(OBS_ENV) + (list(OBS_AUG) if cfg['obs'] == 'env+aug' else [])
    if cfg['no_pest_state']:
        keys.remove('P')
    return keys


def stack_scen(seeds, p):
    scs = [make_scenario(int(s), p) for s in seeds]
    out = {}
    for k, v in scs[0].items():
        if isinstance(v, np.ndarray):
            out[k] = np.stack([sc[k] for sc in scs], 1)           # [time, N]
        else:
            out[k] = np.array([sc[k] for sc in scs], float)
    return out


def vec_init(scb):
    N = len(scb['th0'])
    return dict(th=scb['th0'].copy(), H=scb['RH'][0].copy(), w=np.zeros(N), P=scb['P0'].copy(),
                phi=scb['phi0'].copy(), W=np.zeros(N), E=np.zeros(N), Csoil=np.zeros(N), Epump=np.zeros(N))


def vec_obs(s, t, scb, spec, keys):
    est = dict(th=s['th'], T=scb['T'][t], H=s['H'], P=s['P'])
    return build_obs(est, s, t, scb, spec, keys).T.astype(np.float32)


def vec_rollout(policy_fn, seeds, p, spec, keys, couple=1.0, repeat=1):
    """Deterministic batched rollout (true states, supervisor active). Returns per-scenario metrics."""
    scb = stack_scen(seeds, p); s = vec_init(scb)
    ret = np.zeros(len(seeds)); P = np.zeros(len(seeds))
    for t in range(N_STEPS):
        if t % repeat == 0:
            cmd = policy_fn(vec_obs(s, t, scb, spec, keys))
        u, _ = supervisor(cmd, s['th'], p)
        s, info = step(s, u, t, scb, p, couple=couple)
        r, _ = reward(s, info, p, spec)
        ret += r; P += s['P']
    return dict(ret=ret, water=s['W'], egrid=s['E'], csoil=s['Csoil'], P_mean=P / N_STEPS)


def train(cfg, out_prefix=None, verbose=True):
    cfg = {**DEFAULT_CFG, **cfg}
    rng = np.random.default_rng(cfg['seed']); torch.manual_seed(cfg['seed'])
    p = Params()
    spec = Spec(a1=cfg['a1'], a2=cfg['a2'], a3=cfg['a3'], a4=cfg['a4'], W_budget=cfg['W_budget'],
                E_budget=cfg['E_budget'], d_soil=cfg['d_soil'])
    keys = obs_keys(cfg); od = len(keys); m = 3; NE = cfg['n_env']
    use_costs = cfg['cost_mode'] in ('pacac', 'lag_disc', 'fixed_lam')
    actor = mlp(od, 1); actor_t = copy.deepcopy(actor)
    qr = Ensemble(2, od + 1); qr_t = copy.deepcopy(qr)
    qc = Ensemble(m, od + 1) if use_costs else None; qc_t = copy.deepcopy(qc) if use_costs else None
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg['lr_a'])
    opt_q = torch.optim.Adam(qr.parameters(), lr=cfg['lr_c'])
    opt_c = torch.optim.Adam(qc.parameters(), lr=cfg['lr_c']) if use_costs else None
    lam = np.zeros(m) if cfg['fixed_lam'] is None else np.array(cfg['fixed_lam'], float)
    budgets = np.array([spec.W_budget, spec.E_budget, spec.d_soil])
    E_tot = cfg['rounds'] * NE
    R = cfg['repeat']; TD = N_STEPS // R          # decisions per episode
    gam = cfg['gamma'] ** R                         # per-decision discount (0.99 per 30-min step)
    gc = 1.0 if cfg['cost_mode'] in ('pacac', 'fixed_lam') else gam
    BO = np.zeros((E_tot, TD + 1, od), np.float32); BA = np.zeros((E_tot, TD), np.float32)
    BR = np.zeros((E_tot, TD), np.float32); BC = np.zeros((E_tot, TD, m), np.float32)
    n_done = 0; it = 0
    n = cfg['n_step']; offs = np.arange(n)
    nc = cfg['n_step_c']; offc = np.arange(nc)
    hist, dual_hist = [], []
    best = None; best_ret = None
    t0 = time.time()

    def pol_np(net):
        na = NumpyActor(net, p.u_max)
        return lambda o: na(o)

    for rd in range(cfg['rounds']):
        seeds = 1_000_000 + cfg['seed'] * 10_000 + rd * NE + np.arange(NE)
        scb = stack_scen(seeds, p); s = vec_init(scb)
        e0 = n_done
        ep_r = np.zeros(NE)
        hlo = rng.uniform(0.19, 0.25, NE); hhi = hlo + rng.uniform(0.02, 0.10, NE)
        hrate = rng.uniform(0.2, 0.9, NE); hday = rng.random(NE) < 0.5; hon = np.zeros(NE, bool)
        for t in range(N_STEPS):
            j = t // R
            if t % R == 0:
                o = vec_obs(s, t, scb, spec, keys)
                if rd < cfg['warmup_rounds']:
                    # heuristic-guided exploration: randomised hysteresis rules (one per environment)
                    on = np.where(s['th'] < hlo, True, np.where(s['th'] >= hhi, False, hon))
                    hon = on
                    hr = scb['hour'][t]
                    act = on & (~hday | ((hr >= 8) & (hr <= 17)))
                    a_ = np.where(act, 2 * hrate / p.u_max - 1, -1.0) + rng.normal(0, 0.1, NE)
                    a_ = np.clip(a_, -1, 1)
                else:
                    with torch.no_grad():
                        a_ = torch.tanh(actor(torch.as_tensor(o))).numpy()[:, 0]
                    a_ = np.clip(a_ + rng.normal(0, cfg['expl_sd'], NE), -1, 1)
                BO[e0:e0 + NE, j] = o; BA[e0:e0 + NE, j] = a_
                racc = np.zeros(NE); cacc = np.zeros((NE, m))
            u, _ = supervisor((a_ + 1) / 2 * p.u_max, s['th'], p)
            s1, info = step(s, u, t, scb, p, couple=cfg['couple'])
            r, _ = reward(s1, info, p, spec)
            c = np.stack([info['u'], info['Egrid'], info['c_soil']], 1) / budgets
            if cfg['cost_mode'] == 'penalty':
                r = r - c @ np.asarray(cfg['penalty_kappa'])
            racc += (cfg['gamma'] ** (t % R)) * r; cacc += c
            ep_r += r; s = s1
            if t % R != R - 1:
                continue
            BR[e0:e0 + NE, j] = racc; BC[e0:e0 + NE, j] = cacc
            # ---- gradient updates on completed episodes ----
            if n_done > 0:
                for _ in range(cfg['updates_per_step']):
                    it += 1
                    ei = np.random.randint(0, n_done, cfg['batch']); ti = np.random.randint(0, TD, cfg['batch'])
                    k = np.minimum(n, TD - ti)
                    tj = np.minimum(ti[:, None] + offs[None], TD - 1); mask = offs[None] < k[:, None]
                    O = torch.as_tensor(BO[ei, ti]); A = torch.as_tensor(BA[ei, ti])[:, None]
                    On = torch.as_tensor(BO[ei, ti + k]); Dk = torch.as_tensor((ti + k >= TD).astype(np.float32))
                    Rw = BR[ei[:, None], tj]
                    if cfg['lag_in_reward']:   # r - lambda^T c_bar with the current multipliers
                        Rw = Rw - (BC[ei[:, None], tj] * lam[None, None, :]).sum(-1)
                    Rn = torch.as_tensor((Rw * mask * gam ** offs[None]).sum(1).astype(np.float32))
                    gk = torch.as_tensor((gam ** k).astype(np.float32))
                    OA = torch.cat([O, A], 1)
                    with torch.no_grad():
                        noise = (torch.randn(len(ei), 1) * cfg['tgt_sd']).clamp(-cfg['tgt_clip'], cfg['tgt_clip'])
                        An = (torch.tanh(actor_t(On)) + noise).clamp(-1, 1)
                        y = Rn + gk * (1 - Dk) * qr_t(torch.cat([On, An], 1)).min(0).values
                    lq = ((qr(OA) - y[None]) ** 2).mean()
                    opt_q.zero_grad(); lq.backward(); opt_q.step()
                    if use_costs:
                        with torch.no_grad():
                            kc = np.minimum(nc, TD - ti)
                            tjc = np.minimum(ti[:, None] + offc[None], TD - 1); maskc = offc[None] < kc[:, None]
                            Onc = torch.as_tensor(BO[ei, ti + kc]); Dkc = torch.as_tensor((ti + kc >= TD).astype(np.float32))
                            Cn = torch.as_tensor((BC[ei[:, None], tjc] * (maskc * gc ** offc[None])[:, :, None]).sum(1).astype(np.float32))
                            gck = torch.as_tensor((gc ** kc).astype(np.float32))
                            yc = Cn.T + gck[None] * (1 - Dkc)[None] * qc_t(torch.cat([Onc, torch.tanh(actor_t(Onc))], 1))
                        lc = ((qc(OA) - yc) ** 2).mean()
                        opt_c.zero_grad(); lc.backward(); opt_c.step()
                    if it % cfg['policy_delay'] == 0:
                        pre = actor(O); OAp = torch.cat([O, torch.tanh(pre)], 1)
                        la = -qr(OAp)[0].mean()
                        if use_costs:
                            Qc = qc(OAp).mean(1)
                            if lam.sum() > 0 and not cfg['lag_in_reward']:
                                la = la + (torch.as_tensor(lam, dtype=torch.float32) * Qc).sum()
                            if cfg['cost_mode'] == 'lag_disc':   # original Eq. (30): minibatch critic average
                                lam = np.clip(lam + cfg['lam_lr_step'] * (Qc.detach().numpy() - 1.0), 0, cfg['lam_max'])
                        la = la + cfg['pre_reg'] * (pre ** 2).mean()
                        opt_a.zero_grad(); la.backward(); opt_a.step()
                        with torch.no_grad():
                            for net, tn in [(actor, actor_t), (qr, qr_t)] + ([(qc, qc_t)] if use_costs else []):
                                for w_, wt in zip(net.parameters(), tn.parameters()):
                                    wt.mul_(1 - cfg['tau']).add_(cfg['tau'] * w_)
        BO[e0:e0 + NE, TD] = vec_obs(s, N_STEPS - 1, scb, spec, keys)   # terminal obs (unused: done)
        n_done += NE
        # ---- deterministic Monte-Carlo estimate of constraint returns (fresh training scenarios) ----
        dseeds = 2_000_000 + cfg['seed'] * 10_000 + rd * cfg['n_dual'] + np.arange(cfg['n_dual'])
        me = vec_rollout(pol_np(actor), dseeds, p, spec, keys, cfg['couple'], R)
        cdet = np.array([me['water'].mean(), me['egrid'].mean(), me['csoil'].mean()]) / budgets
        if rd >= cfg['warmup_rounds']:
            dual_hist.append(cdet)
        if cfg['cost_mode'] == 'pacac' and rd >= cfg['warmup_rounds']:
            Jmc = np.mean(dual_hist[-cfg['lam_window']:], 0)
            lam = np.clip(lam + cfg['lam_lr'] * (Jmc - 1.0), 0, cfg['lam_max'])
        q0 = [np.nan] * m
        if use_costs:
            sc0 = stack_scen(dseeds[:1], p); s0 = vec_init(sc0)
            o0 = torch.as_tensor(vec_obs(s0, 0, sc0, spec, keys))
            with torch.no_grad():
                q0 = qc(torch.cat([o0, torch.tanh(actor(o0))], 1))[:, 0].numpy().tolist()
        h = dict(round=rd, episodes=n_done, train_ret=float(ep_r.mean()), lam=lam.tolist(), q0=q0,
                 det={k2: float(np.mean(v)) for k2, v in me.items()}, time=time.time() - t0, updates=it)
        if rd >= cfg['warmup_rounds'] and (rd % cfg['ckpt_every'] == cfg['ckpt_every'] - 1 or rd == cfg['rounds'] - 1):
            dv = vec_rollout(pol_np(actor), 5000 + np.arange(cfg['n_dev']), p, spec, keys, cfg['couple'], R)
            dv = {k2: float(np.mean(v)) for k2, v in dv.items()}
            viol = max(dv['water'] / spec.W_budget, dv['egrid'] / spec.E_budget, dv['csoil'] / spec.d_soil, 1.0) - 1.0
            score = (1, viol) if viol > 0 else (0, -dv['ret'])
            h['dev'] = dict(dv, viol=viol)
            if cfg['cost_mode'] == 'none':          # unconstrained learner: select on return only
                score = (0, -dv['ret'])
            if best is None or score < best[0]:
                best = (score, copy.deepcopy(actor), rd)
            if best_ret is None or dv['ret'] > best_ret[0]:
                best_ret = (dv['ret'], copy.deepcopy(actor), rd)
        hist.append(h)
        if verbose:
            print(f"[{cfg.get('name','')} s{cfg['seed']}] rd{rd} ep{n_done} train={ep_r.mean():.0f} | det ret={h['det']['ret']:.0f} "
                  f"W={h['det']['water']:.1f} E={h['det']['egrid']:.2f} S={h['det']['csoil']:.3f} P={h['det']['P_mean']:.3f} "
                  f"lam={np.round(lam,2)} q0={np.round(q0,2)} dev={h.get('dev',{}).get('ret',float('nan')):.0f} t={h['time']:.0f}s", flush=True)
    na = NumpyActor(best[1] if best else actor, p.u_max)
    cfg['selected_round'] = best[2] if best else None
    cfg['selected_round_return'] = best_ret[2] if best_ret else None
    if out_prefix:
        na.save(out_prefix + '_actor.npz')
        if best_ret:
            NumpyActor(best_ret[1], p.u_max).save(out_prefix + '_actorret.npz')
        json.dump(dict(cfg=cfg, keys=keys, hist=hist), open(out_prefix + '_hist.json', 'w'))
    return na, keys, hist, cfg
