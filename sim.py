"""
Fully specified stochastic agro-environmental testbed for PAC-AC (revised manuscript).

Lumped (single-bucket) root-zone water balance, FAO-56 crop ET with water-stress
coefficient, Markov-chain/gamma rainfall generator, canopy humidity and leaf-wetness
dynamics with sprinkler wetting, a generic foliar-fungal infection-favourability index
(Magarey et al. 2005 temperature response) with Poisson evidence, NDVI evolution,
and PV-assisted pumping energy. All exogenous randomness is pre-generated per
scenario so that controllers are evaluated on identical disturbance sequences
(action-independent random-number consumption).

All arrays support a leading batch dimension (used by the MPC planner).
"""
import numpy as np
from dataclasses import dataclass, field, asdict, replace

DT_H = 0.5                    # control interval [h]
STEPS_PER_DAY = 48
N_DAYS = 14
N_STEPS = N_DAYS * STEPS_PER_DAY   # 672 decisions per rollout, t = 0..N-1
FC_STEPS = 24                 # 12-h rainfall-forecast window


@dataclass
class Params:
    # --- soil (sandy loam, lumped root zone) ---
    Zr: float = 400.0          # effective root-zone depth [mm]
    th_s: float = 0.41         # saturation VWC [m3/m3]
    th_fc: float = 0.30        # field capacity
    th_wp: float = 0.12        # wilting point
    p_dep: float = 0.40        # FAO-56 depletion fraction (tomato)
    k_d: float = 0.04          # drainage coefficient for water above FC [1/step]
    # --- agronomic limits / target ---
    th_min: float = 0.18
    th_max: float = 0.34
    th_star: float = 0.25
    d_theta: float = 0.08      # half-width of crop-proxy band
    # --- crop / ET ---
    Kc: float = 1.15           # FAO-56 mid-season tomato
    Ra: float = 15.5           # extraterrestrial radiation [mm/day equiv.], ~27 deg N, Aug
    # --- irrigation (micro-sprinkler) and pumping ---
    u_max: float = 0.9         # [mm/step] = 2.5 L/s on 0.5 ha
    area: float = 5000.0       # [m2]
    head: float = 30.0         # total dynamic head [m]
    eta_p: float = 0.50        # wire-to-water pump efficiency
    pv_kwp: float = 1.5        # PV array peak [kW]
    loss_a: float = 0.04       # sprinkler evaporation/drift loss: a + b*VPD + c*U
    loss_b: float = 0.03
    loss_c: float = 0.01
    # --- canopy microclimate ---
    kappa_h: float = 0.5       # relaxation of canopy RH to background [1/step]
    beta_th: float = 40.0      # soil-moisture effect on canopy RH [%RH per VWC]
    beta_u: float = 12.0       # sprinkler humidification [%RH per mm, scaled by (1-H/100)]
    beta_R: float = 8.0        # rainfall humidification [%RH per mm]
    sig_h: float = 1.5         # process noise [%RH]
    # --- leaf wetness ---
    a_u: float = 0.9           # wetting by sprinkler (u >= u_wet wets canopy fully)
    u_wet: float = 0.3         # [mm/step]
    a_R: float = 0.9
    a_dew: float = 0.3
    H_dew: float = 92.0
    k_dry: float = 0.25        # drying per kPa VPD per step
    k_dry_U: float = 0.05      # drying per m/s wind per step
    # --- foliar-disease favourability index ---
    Tmin_p: float = 10.0       # Magarey et al. (2005) cardinal temperatures
    Topt_p: float = 24.0
    Tmax_p: float = 34.0
    eta_p_inf: float = 0.04    # accumulation rate
    delta_p: float = 0.010     # decay [1/step]
    lam0: float = 0.3          # Poisson evidence intensity scale [events/step]
    jump: float = 0.03         # index increment per evidence event
    # --- NDVI ---
    phi_max: float = 0.85
    g_phi: float = 0.05        # recovery [1/day]
    s_w: float = 0.03          # water-stress decline [1/day]
    s_p: float = 0.04          # disease decline [1/day per unit (P-0.5)+]
    # --- weather generator ---
    p_wd: float = 0.15         # P(wet | previous dry)
    p_ww: float = 0.40         # P(wet | previous wet)
    g_shape: float = 0.75      # gamma shape of wet-day rainfall
    g_scale: float = 12.0      # gamma scale [mm]
    Tmax_mu: float = 33.0
    Tmax_sd: float = 2.0
    Tmin_mu: float = 22.0
    Tmin_sd: float = 1.8
    wet_cool: float = 3.0      # Tmax reduction on wet days
    RH_mu: float = 64.0
    RH_sd: float = 7.0
    RH_wet: float = 12.0
    RH_amp: float = 18.0
    U_mu: float = 2.0
    U_sd: float = 0.6
    # --- forecast ---
    fc_sd: float = 0.5         # lognormal sd of 12-h rain forecast error
    fc_miss: float = 0.2       # probability that a day's rain is missed by the forecast
    # --- scenario multipliers for sensitivity tests ---
    rain_mult: float = 1.0
    T_shift: float = 0.0


def solar_shape(hour):
    """Normalised diurnal radiation shape (zero at night)."""
    return np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))


def es_kpa(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


def magarey_f(T, p: Params):
    """Generic temperature response for foliar fungal infection (Magarey et al. 2005)."""
    Tn, To, Tx = p.Tmin_p, p.Topt_p, p.Tmax_p
    Tc = np.clip(T, Tn, Tx)
    f = ((Tx - Tc) / (Tx - To)) * ((Tc - Tn) / (To - Tn)) ** ((To - Tn) / (Tx - To))
    return np.clip(f, 0.0, 1.0)


def make_scenario(seed: int, p: Params = Params()):
    """Pre-generate every exogenous random quantity of one 14-day rollout."""
    rng = np.random.default_rng(seed)
    nd = N_DAYS + 1  # one extra day for forecast look-ahead
    wet = np.zeros(nd, bool)
    wet[0] = rng.random() < p.p_wd / (1 + p.p_wd - p.p_ww)
    for d in range(1, nd):
        wet[d] = rng.random() < (p.p_ww if wet[d - 1] else p.p_wd)
    amount = np.where(wet, rng.gamma(p.g_shape, p.g_scale, nd), 0.0) * p.rain_mult
    start = rng.uniform(12.0, 22.0, nd)
    dur = rng.uniform(2.0, 6.0, nd)
    Tmax = rng.normal(p.Tmax_mu, p.Tmax_sd, nd) - p.wet_cool * wet + p.T_shift
    Tmin = rng.normal(p.Tmin_mu, p.Tmin_sd, nd) + p.T_shift
    Tmin = np.minimum(Tmin, Tmax - 4.0)
    RHm = rng.normal(p.RH_mu, p.RH_sd, nd) + p.RH_wet * wet
    Um = np.maximum(0.3, rng.normal(p.U_mu, p.U_sd, nd))
    cloud = np.where(wet, 0.45, 0.9) * rng.uniform(0.85, 1.0, nd)

    n = nd * STEPS_PER_DAY
    tt = np.arange(n)
    day = tt // STEPS_PER_DAY
    hour = (tt % STEPS_PER_DAY) * DT_H + DT_H / 2
    Tmean = 0.5 * (Tmax + Tmin)
    T = Tmean[day] + 0.5 * (Tmax - Tmin)[day] * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
    RH = np.clip(RHm[day] - p.RH_amp * np.cos(2 * np.pi * (hour - 15.0) / 24.0), 30, 98)
    U = Um[day] * (0.6 + 0.8 * solar_shape(hour))
    # rainfall: uniform intensity storm inside the day
    in_storm = (hour >= start[day]) & (hour < np.minimum(start[day] + dur[day], 24.0))
    n_storm = np.array([max(1, int(np.sum(in_storm[day == d]))) for d in range(nd)])
    rain = np.where(in_storm & wet[day], amount[day] / n_storm[day], 0.0)
    # Hargreaves-Samani ET0 [mm/day] distributed by the radiation shape
    et0_day = 0.0023 * p.Ra * (Tmean + 17.8) * np.sqrt(np.maximum(Tmax - Tmin, 0.1))
    et0_day = et0_day * np.where(wet, 0.7, 1.0)
    shape = solar_shape(hour)
    shape_sum = solar_shape((np.arange(STEPS_PER_DAY) * DT_H + DT_H / 2)).sum()
    et0 = et0_day[day] * shape / shape_sum
    pv = p.pv_kwp * shape * cloud[day] * DT_H        # [kWh/step]
    # noisy 12-h rainfall forecast (day-level multiplicative error + misses)
    fc_err = np.exp(rng.normal(0, p.fc_sd, nd)) * (rng.random(nd) > p.fc_miss)
    csum = np.concatenate([[0.0], np.cumsum(rain)])
    true_fc = np.array([csum[min(t + 1 + FC_STEPS, n)] - csum[min(t + 1, n)] for t in range(n)])
    rain_fc = true_fc * fc_err[day]
    sc = dict(
        T=T, RH=RH, U=U, rain=rain, et0=et0, pv=pv, rain_fc=rain_fc, hour=hour,
        u_pois=rng.random(n),                 # uniforms for Poisson evidence (inverse CDF)
        xi_h=rng.normal(0, 1, n),             # canopy-humidity process noise
        n_th=rng.normal(0, 1, n), n_T=rng.normal(0, 1, n), n_H=rng.normal(0, 1, n),
        drop=rng.random(n),                   # sensor-dropout uniforms
        th0=rng.uniform(0.22, 0.28), P0=rng.uniform(0.05, 0.15), phi0=rng.uniform(0.60, 0.70),
        wet_days=wet[:N_DAYS].sum(), rain_total=rain[:N_STEPS].sum(),
    )
    return sc


def poisson_inv(u, lam):
    """Inverse-CDF Poisson draw (vectorised, small lam)."""
    lam = np.asarray(lam, float)
    k = np.zeros_like(lam)
    pk = np.exp(-lam)
    cdf = pk.copy()
    for i in range(1, 8):
        more = u > cdf
        k = k + more
        pk = pk * lam / i
        cdf = cdf + pk
    return k


def init_state(sc, batch=None):
    s = dict(th=sc['th0'], H=sc['RH'][0], w=0.0, P=sc['P0'], phi=sc['phi0'],
             W=0.0, E=0.0, Csoil=0.0, Epump=0.0)
    if batch is not None:
        s = {k: np.full(batch, v, float) for k, v in s.items()}
    return s


def ks_coef(th, p: Params):
    th_p = p.th_fc - p.p_dep * (p.th_fc - p.th_wp)
    return np.clip((th - p.th_wp) / (th_p - p.th_wp), 0.0, 1.0)


def step(s, u, t, sc, p: Params, couple=1.0, stochastic=True):
    """One 30-min transition. s: dict of arrays/floats, u: applied irrigation [mm].
    Returns next state dict and info (costs, components)."""
    T, RH, U, R = sc['T'][t], sc['RH'][t], sc['U'][t], sc['rain'][t]
    th, H, w, P, phi = s['th'], s['H'], s['w'], s['P'], s['phi']
    vpd = es_kpa(T) * (1 - H / 100.0)
    # --- soil water balance (Eq. 2 of the manuscript) ---
    loss = np.clip(p.loss_a + p.loss_b * vpd + p.loss_c * U, 0.0, 0.3)
    u_inf = u * (1 - loss)
    Ks = ks_coef(th, p)
    etc = Ks * p.Kc * sc['et0'][t]
    D = p.k_d * np.maximum(th - p.th_fc, 0.0) * p.Zr
    th1 = th + (u_inf + R - etc - D) / p.Zr
    runoff = np.maximum(th1 - p.th_s, 0.0) * p.Zr
    th1 = np.minimum(th1, p.th_s)
    # --- canopy humidity f_h ---
    xi = sc['xi_h'][t] * p.sig_h if stochastic else 0.0
    RHn = sc['RH'][min(t + 1, len(sc['RH']) - 1)]
    H1 = H + p.kappa_h * (RHn + p.beta_th * (th - p.th_star) - H) \
        + (couple * p.beta_u * u + p.beta_R * R) * (1 - H / 100.0) + xi
    H1 = np.clip(H1, 20.0, 100.0)
    # --- leaf wetness ---
    vpd1 = es_kpa(T) * (1 - H1 / 100.0)
    w1 = w + couple * p.a_u * np.minimum(u / p.u_wet, 1.0) * (1 - w) \
        + p.a_R * np.minimum(R / 1.0, 1.0) * (1 - w) + p.a_dew * (H1 >= p.H_dew) * (1 - w)
    w1 = np.clip(w1, 0, 1)
    w1 = np.clip(w1 * (1 - np.minimum(1.0, p.k_dry * vpd1 + p.k_dry_U * U)), 0.0, 1.0)
    # --- disease favourability index f_p ---
    iota = w1 * magarey_f(T, p) * (0.5 + 0.5 * phi / p.phi_max)
    if stochastic:
        N_ev = poisson_inv(sc['u_pois'][t], p.lam0 * iota)
    else:
        N_ev = p.lam0 * iota
    P1 = (1 - p.delta_p) * P + p.eta_p_inf * iota * (1 - P) + p.jump * N_ev * (1 - P)
    P1 = np.clip(P1, 0.0, 1.0)
    # --- NDVI ---
    phi1 = phi + (p.g_phi * (p.phi_max - phi) * Ks - p.s_w * (1 - Ks)
                  - p.s_p * np.maximum(P - 0.5, 0.0)) / STEPS_PER_DAY
    phi1 = np.clip(phi1, 0.1, p.phi_max)
    # --- energy ---
    V = u * p.area / 1000.0                                    # m3
    Epump = 1000 * 9.81 * p.head * V / p.eta_p / 3.6e6         # kWh
    Egrid = np.maximum(Epump - sc['pv'][t], 0.0)
    c_soil = np.maximum(th1 - p.th_max, 0.0) + np.maximum(p.th_min - th1, 0.0)
    s1 = dict(th=th1, H=H1, w=w1, P=P1, phi=phi1, W=s['W'] + u, E=s['E'] + Egrid,
              Csoil=s['Csoil'] + c_soil, Epump=s['Epump'] + Epump)
    info = dict(u=u, Egrid=Egrid, Epump=Epump, c_soil=c_soil, Ks=Ks, etc=etc, D=D,
                runoff=runoff, loss=u * loss, rain=R, iota=iota, vpd=vpd)
    return s1, info


# ---------------- reward / constraint specification ----------------
@dataclass
class Spec:
    a1: float = 1.0     # crop proxy
    a2: float = 0.3     # water   (normalised by u_max)
    a3: float = 0.3     # grid energy (normalised by max step pump energy)
    a4: float = 1.0     # disease index
    W_budget: float = 38.0    # [mm per 14-day rollout]
    E_budget: float = 6.0    # [kWh grid energy per rollout]
    d_soil: float = 0.15      # [VWC-steps] cumulative soil-limit violation


def reward(s1, info, p: Params, spec: Spec):
    ry = np.maximum(0.0, 1 - ((s1['th'] - p.th_star) / p.d_theta) ** 2)
    e_max = 1000 * 9.81 * p.head * p.u_max * p.area / 1000 / p.eta_p / 3.6e6
    r = spec.a1 * ry - spec.a2 * info['u'] / p.u_max - spec.a3 * info['Egrid'] / e_max \
        - spec.a4 * s1['P']
    return r, ry


# ---------------- safety supervisor (identical for every controller) ----------------
SUP_SAT_MARGIN = 0.02
SUP_DEF_MARGIN = 0.01
SUP_DEF_RATE = 0.3        # maintenance irrigation applied by the deficit guard [mm/step]


def supervisor(u_cmd, th_hat, p: Params):
    """S1 actuator bounds; S2 saturation guard (priority); S3 deficit guard.
    Returns executed action and intervention code (0 none, 1 sat-guard, 2 deficit-guard)."""
    u = np.clip(u_cmd, 0.0, p.u_max)
    code = np.zeros_like(np.asarray(u, float))
    sat = th_hat >= p.th_max - SUP_SAT_MARGIN
    dfc = (th_hat <= p.th_min + SUP_DEF_MARGIN) & ~sat
    u2 = np.where(sat, 0.0, np.where(dfc, np.maximum(u, SUP_DEF_RATE), u))
    code = np.where(sat & (u > 1e-3), 1, np.where(dfc & (u < SUP_DEF_RATE - 1e-3), 2, 0))
    return u2, code


# ---------------- observation model ----------------
@dataclass
class Sensing:
    noisy: bool = False
    sd_th: float = 0.015
    sd_T: float = 0.5
    sd_H: float = 3.0
    bias_th: float = 0.0
    p_drop: float = 0.0
    use_kf: bool = True
    q_kf: float = 2e-5     # KF process variance [VWC^2/step]
    pest_misspec: float = 0.0   # Topt shift used by the edge risk model


class Estimator:
    """Edge state estimator used in the robustness experiments. For the lumped
    water balance the theta process model is linear given inputs, so the UKF of the
    deployment design reduces to a scalar Kalman filter; T and RH use EWMA; the
    disease index is recomputed at the edge from estimated microclimate."""

    def __init__(self, sc, p: Params, sen: Sensing):
        self.p, self.sen, self.sc = p, sen, sc
        self.th = sc['th0'] + sen.bias_th
        self.Pv = sen.sd_th ** 2
        self.Tf, self.Hf = None, None
        self.P_edge = sc['P0']
        self.pm = replace(p, Topt_p=p.Topt_p + sen.pest_misspec)

    def update(self, t, s_true, u_prev, info_prev):
        sen, p, sc = self.sen, self.p, self.sc
        drop = sc['drop'][t] < sen.p_drop
        z_th = s_true['th'] + sen.bias_th + (sc['n_th'][t] * sen.sd_th if sen.noisy else 0)
        z_T = sc['T'][t] + (sc['n_T'][t] * sen.sd_T if sen.noisy else 0)
        z_H = s_true['H'] + (sc['n_H'][t] * sen.sd_H if sen.noisy else 0)
        if not sen.noisy:
            self.th, self.Tf, self.Hf = s_true['th'], z_T, z_H
        else:
            if t > 0:   # KF predict with nominal water balance using logged inputs
                Ks = ks_coef(self.th, p)
                pred = self.th + (info_prev['u'] - info_prev['loss'] + info_prev['rain']
                                  - Ks * p.Kc * sc['et0'][t - 1]
                                  - p.k_d * max(self.th - p.th_fc, 0) * p.Zr) / p.Zr
                self.th, self.Pv = pred, self.Pv + sen.q_kf
            if sen.use_kf:
                if not drop:
                    K = self.Pv / (self.Pv + sen.sd_th ** 2)
                    self.th = self.th + K * (z_th - self.th)
                    self.Pv = (1 - K) * self.Pv
            else:
                if not drop:
                    self.th = z_th
            a = 0.5
            self.Tf = z_T if self.Tf is None else a * z_T + (1 - a) * self.Tf
            self.Hf = z_H if self.Hf is None else a * z_H + (1 - a) * self.Hf
        # edge disease index (possibly misspecified model)
        if sen.pest_misspec != 0.0 or sen.noisy:
            iota = s_true['w'] * magarey_f(self.Tf, self.pm) * (0.5 + 0.5 * s_true['phi'] / p.phi_max)
            self.P_edge = (1 - p.delta_p) * self.P_edge + p.eta_p_inf * iota * (1 - self.P_edge)
            P_hat = self.P_edge
        else:
            P_hat = s_true['P']
        return dict(th=self.th, T=self.Tf, H=self.Hf, P=P_hat)


OBS_ENV = ['th', 'T', 'H', 'R', 'Rfc', 'phi', 'P', 'w', 'pv', 'sin', 'cos']
OBS_AUG = ['tau', 'bw', 'be', 'bs']


def build_obs(est, s, t, sc, spec: Spec, keys):
    hour = sc['hour'][t]
    feats = dict(
        th=(est['th'] - 0.25) / 0.05, T=(est['T'] - 28) / 5, H=(est['H'] - 65) / 15,
        R=sc['rain'][t] / 2.0, Rfc=sc['rain_fc'][t] / 10.0, phi=(s['phi'] - 0.7) / 0.1,
        P=(est['P'] - 0.3) / 0.2, w=s['w'], pv=sc['pv'][t] / (Params().pv_kwp * DT_H), sin=np.sin(2 * np.pi * hour / 24), cos=np.cos(2 * np.pi * hour / 24),
        tau=1 - t / N_STEPS, bw=1 - s['W'] / spec.W_budget, be=1 - s['E'] / spec.E_budget,
        bs=1 - s['Csoil'] / spec.d_soil)
    return np.array(np.broadcast_arrays(*[feats[k] for k in keys]), dtype=np.float32)


def rollout(controller, sc, p: Params = Params(), spec: Spec = Spec(), sen: Sensing = Sensing(),
            couple=1.0, use_supervisor=True, log=False):
    """Evaluate one controller on one pre-generated scenario. Returns metrics (and log)."""
    s = init_state(sc)
    est_model = Estimator(sc, p, sen)
    controller.reset(sc, p, spec)
    info = None
    rec = {k: [] for k in ['th', 'th_hat', 'H', 'w', 'P', 'phi', 'u_cmd', 'u', 'code', 'Egrid',
                           'Epump', 'c_soil', 'rain', 'r', 'ry', 'Ks', 'loss', 'T', 'pv', 'drain']} if log else None
    tot = dict(r=0.0, ry=0.0, err=0.0, Ks=0.0, sup=0, sup_sat=0, sup_def=0, sup_mag=0.0,
               hi_irr=0, hi_steps=0, viol_steps=0, Ppeak=0.0, Psum=0.0, loss=0.0, drain=0.0, runoff=0.0)
    for t in range(N_STEPS):
        est = est_model.update(t, s, None, info)
        u_cmd = float(controller.act(est, s, t, sc))
        if use_supervisor:
            u, code = supervisor(u_cmd, est['th'], p)
            u, code = float(u), int(code)
        else:
            u, code = float(np.clip(u_cmd, 0, p.u_max)), 0
        P_now = s['P']
        s1, info = step(s, u, t, sc, p, couple=couple)
        r, ry = reward(s1, info, p, spec)
        tot['r'] += r; tot['ry'] += ry
        tot['err'] += abs(s1['th'] - p.th_star); tot['Ks'] += info['Ks']
        tot['sup'] += code > 0; tot['sup_sat'] += code == 1; tot['sup_def'] += code == 2
        tot['sup_mag'] += abs(u - np.clip(u_cmd, 0, p.u_max))
        tot['hi_irr'] += (P_now > 0.5) and (u > 0.05); tot['hi_steps'] += P_now > 0.5
        tot['viol_steps'] += info['c_soil'] > 0
        tot['Ppeak'] = max(tot['Ppeak'], s1['P']); tot['Psum'] += s1['P']
        tot['loss'] += info['loss']; tot['drain'] += info['D']; tot['runoff'] += info['runoff']
        if log:
            for k, v in [('th', s['th']), ('th_hat', est['th']), ('H', s['H']), ('w', s['w']), ('P', s['P']),
                         ('phi', s['phi']), ('u_cmd', u_cmd), ('u', u), ('code', code), ('Egrid', info['Egrid']),
                         ('Epump', info['Epump']), ('c_soil', info['c_soil']), ('rain', info['rain']), ('r', r),
                         ('ry', ry), ('Ks', info['Ks']), ('loss', info['loss']), ('T', sc['T'][t]),
                         ('pv', sc['pv'][t]), ('drain', info['D'])]:
                rec[k].append(float(v))
        controller.observe(u, info)
        s = s1
    m = dict(water=s['W'], egrid=s['E'], epump=s['Epump'], csoil=s['Csoil'], ret=tot['r'],
             crop_proxy=tot['ry'] / N_STEPS, mae_theta=tot['err'] / N_STEPS, rel_transp=tot['Ks'] / N_STEPS,
             P_mean=tot['Psum'] / N_STEPS, P_peak=tot['Ppeak'], hi_irr=tot['hi_irr'], hi_steps=tot['hi_steps'],
             sup=tot['sup'], sup_sat=tot['sup_sat'], sup_def=tot['sup_def'], sup_mag=tot['sup_mag'],
             viol_steps=tot['viol_steps'], loss=tot['loss'], drain=tot['drain'], runoff=tot['runoff'],
             phi_end=s['phi'], th_end=s['th'], rain=sc['rain_total'],
             w_exceed=float(s['W'] > spec.W_budget), e_exceed=float(s['E'] > spec.E_budget),
             s_exceed=float(s['Csoil'] > spec.d_soil))
    if log:
        return m, {k: np.array(v) for k, v in rec.items()}
    return m
