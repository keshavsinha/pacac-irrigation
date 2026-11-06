import numpy as np
from sim import (Params, Spec, N_STEPS, FC_STEPS, step, reward, init_state, supervisor, build_obs,
                 ks_coef, es_kpa, magarey_f, STEPS_PER_DAY)


class Controller:
    def reset(self, sc, p, spec):
        self.p, self.spec, self.sc = p, spec, sc

    def act(self, est, s, t, sc):
        raise NotImplementedError

    def observe(self, u, info):
        pass


class Threshold(Controller):
    """Hysteresis controller: ON below th_low, OFF at/above th_high, ON-rate q [mm/step]."""

    def __init__(self, th_low=0.20, th_high=0.30, rate=0.9, day_only=False):
        self.lo, self.hi, self.rate, self.day_only = th_low, th_high, rate, day_only

    def reset(self, sc, p, spec):
        super().reset(sc, p, spec)
        self.on = False

    def act(self, est, s, t, sc):
        th = est['th']
        if th < self.lo:
            self.on = True
        elif th >= self.hi:
            self.on = False
        if self.day_only:
            h = sc['hour'][t]
            if not (8 <= h <= 17):
                return 0.0
        return self.rate if self.on else 0.0


class Policy(Controller):
    """Wraps a trained actor (numpy forward pass)."""

    def __init__(self, actor_np, keys, spec_for_obs=None, repeat=1):
        self.actor, self.keys = actor_np, keys
        self.spec_obs = spec_for_obs
        self.repeat = repeat
        self.cmd = 0.0

    def act(self, est, s, t, sc):
        if t % self.repeat == 0:
            o = build_obs(est, s, t, sc, self.spec_obs or self.spec, self.keys)
            self.cmd = float(self.actor(o))
        return self.cmd


class MPC(Controller):
    """Certainty-equivalent receding-horizon controller solved by the cross-entropy
    method. Uses the nominal simulator structure, the same noisy 12-h rainfall forecast
    as the learning controllers, persistence of the current-day microclimate profile
    perturbed by forecast error, expected (mean) disease evidence, the same scalar
    reward, a soil-limit penalty, and a water/energy shadow price (tuned on development
    scenarios) for the 14-day budgets. Horizon 24 steps, 6 blocks of 4 steps."""

    def __init__(self, H=24, blocks=6, pop=64, elite=8, iters=3, lam_w=0.0, lam_e=0.0,
                 rho_soil=200.0, seed=0, term_w=2.0):
        self.Hh, self.nb, self.pop, self.elite, self.iters = H, blocks, pop, elite, iters
        self.lam_w, self.lam_e, self.rho = lam_w, lam_e, rho_soil
        self.rng = np.random.default_rng(seed)
        self.term_w = term_w

    def reset(self, sc, p, spec):
        super().reset(sc, p, spec)
        self.mu = np.full(self.nb, 0.2 * p.u_max)
        self.rng = np.random.default_rng(12345)
        # forecast error of microclimate (per-rollout, pre-drawn)
        frng = np.random.default_rng(int(sc['th0'] * 1e6) % (2 ** 31))
        self.dT = frng.normal(0, 1.0, N_STEPS + FC_STEPS + 48)
        self.dH = frng.normal(0, 5.0, N_STEPS + FC_STEPS + 48)

    def _fsc(self, t, sc):
        """Build forecast scenario arrays for steps t..t+H."""
        idx = np.arange(t, t + self.Hh)
        idx = np.minimum(idx, len(sc['T']) - 2)
        fsc = {k: sc[k][idx] for k in ['T', 'RH', 'U', 'et0', 'pv', 'hour']}
        fsc['T'] = fsc['T'] + self.dT[idx]
        fsc['RH'] = np.clip(fsc['RH'] + self.dH[idx], 30, 98)
        fsc['RH'] = np.concatenate([fsc['RH'], fsc['RH'][-1:]])
        rf = sc['rain_fc'][t]
        fsc['rain'] = np.full(self.Hh, rf / FC_STEPS)
        fsc['xi_h'] = np.zeros(self.Hh)
        fsc['u_pois'] = np.zeros(self.Hh)
        return fsc

    def act(self, est, s, t, sc):
        p, spec = self.p, self.spec
        fsc = self._fsc(t, sc)
        H = min(self.Hh, N_STEPS - t)
        bl = self.Hh // self.nb
        mu, sd = self.mu.copy(), np.full(self.nb, 0.35 * p.u_max)
        best, best_J = mu.copy(), -np.inf
        for it in range(self.iters):
            A = np.clip(mu + sd * self.rng.standard_normal((self.pop, self.nb)), 0, p.u_max)
            A[0] = np.clip(mu, 0, p.u_max)
            A[1] = 0.0
            J = self._evaluate(A, est, s, fsc, H, bl)
            el = A[np.argsort(-J)[:self.elite]]
            if J.max() > best_J:
                best_J, best = J.max(), A[np.argmax(J)].copy()
            mu, sd = el.mean(0), el.std(0) + 0.02
        self.mu = np.concatenate([best[1:], best[-1:]]) if t % bl == bl - 1 else best
        return float(best[0])

    def _evaluate(self, A, est, s, fsc, H, bl):
        p, spec = self.p, self.spec
        n = A.shape[0]
        st = dict(th=np.full(n, est['th']), H=np.full(n, est['H']), w=np.full(n, s['w']),
                  P=np.full(n, est['P']), phi=np.full(n, s['phi']), W=np.zeros(n), E=np.zeros(n),
                  Csoil=np.zeros(n), Epump=np.zeros(n))
        J = np.zeros(n)
        for k in range(H):
            u = A[:, min(k // bl, self.nb - 1)]
            u = np.where(st['th'] >= p.th_max - 0.02, 0.0, u)
            st, info = step(st, u, k, fsc, p, stochastic=False)
            r, _ = reward(st, info, p, spec)
            J += r - self.rho * info['c_soil'] - self.lam_w * u - self.lam_e * info['Egrid']
        # terminal value: penalise ending below the target (avoid myopic depletion)
        J -= self.term_w * 48 * np.maximum(p.th_star - st['th'], 0) / p.d_theta
        return J
