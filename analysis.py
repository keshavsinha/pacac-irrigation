"""Statistics, LaTeX tables and figures for the revised manuscript."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
LAB = {'thr_classic': 'Threshold (classical)', 'thr_tuned': 'Threshold (tuned)', 'mpc': 'MPC',
       'td3': 'TD3', 'td3lag': 'TD3-Lag', 'pacac': 'PAC-AC',
       'pacac_noaug': 'PAC-AC w/o time/budget features', 'pacac_nopstate': 'PAC-AC w/o disease-index feature',
       'pacac_noprew': 'PAC-AC w/o disease reward', 'pacac_nopest': 'PAC-AC w/o disease feature and reward',
       'pacac_penalty': 'Fixed multipliers (no dual update)', 'pacac_cgrad': 'PAC-AC, cost-critic gradient', 'pacac_fixlam': 'Fixed multipliers',
       'sens_a4x0.5': r'$\alpha_4\times0.5$', 'sens_a4x2': r'$\alpha_4\times2$', 'sens_a2x2': r'$\alpha_2\times2$',
       'sens_Wx0.85': r'$W_b\times0.85$', 'sens_Wx1.15': r'$W_b\times1.15$'}
COL = {'thr_classic': '#d62728', 'thr_tuned': '#ff7f0e', 'mpc': '#9467bd', 'td3': '#2ca02c',
       'td3lag': '#8c564b', 'pacac': '#1f77b4'}
METRICS = [  # key, label, unit, fmt
    ('water', 'Applied water', 'mm/14 d', 1), ('egrid', 'Grid energy', 'kWh/14 d', 2),
    ('epump', 'Pumping energy', 'kWh/14 d', 1), ('csoil', 'Soil-limit violation', r'VWC$\cdot$step', 3),
    ('viol_steps', 'Steps outside limits', 'steps', 1), ('mae_theta', r'Mean $|\theta-\theta^*|$', 'VWC', 4),
    ('crop_proxy', r'Crop proxy $\bar r^{(y)}$', '--', 3), ('rel_transp', r'Rel.\ transpiration $\bar K_s$', '--', 3),
    ('P_mean', 'Mean disease index', '--', 3), ('P_peak', 'Peak disease index', '--', 3),
    ('hi_steps', r'Steps with $P_t>0.5$', 'steps', 1), ('hi_irr', r'Irrigated steps with $P_t>0.5$', 'steps', 1),
    ('sup', 'Supervisor interventions', 'steps', 1), ('sup_mag', 'Corrected depth', 'mm', 2),
    ('ret', 'Return (undiscounted)', '--', 1)]
SECONDARY = ['egrid', 'csoil', 'mae_theta', 'crop_proxy', 'rel_transp', 'P_mean', 'P_peak', 'hi_steps', 'hi_irr', 'sup', 'ret']


def load(cond='nominal'):
    dfs = [pd.read_csv(f) for f in glob.glob('results/eval/*.csv')]
    df = pd.concat(dfs)
    return df[df.cond == cond]


def per_scenario(df, method):
    d = df[df.method == method]
    return d.groupby('scenario').mean(numeric_only=True).sort_index()


def seed_sd(df, method, key):
    d = df[df.method == method]
    if d.train_seed.nunique() < 2:
        return np.nan
    return d.groupby('train_seed')[key].mean().std(ddof=1)


def paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    n = len(d); m = d.mean(); sd = d.std(ddof=1)
    if sd == 0:
        return dict(diff=m, lo=m, hi=m, t=np.nan, p=1.0 if m == 0 else 0.0, dz=np.nan, pw=1.0, n=n)
    se = sd / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    t = m / se; p = 2 * stats.t.sf(abs(t), n - 1)
    try:
        pw = stats.wilcoxon(d).pvalue
    except ValueError:
        pw = 1.0
    return dict(diff=m, lo=m - tc * se, hi=m + tc * se, t=t, p=p, dz=m / sd, pw=pw, n=n,
                shapiro=stats.shapiro(d).pvalue if n >= 3 else np.nan)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); k = len(ps)
    adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, (k - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def fmt(v, nd):
    if pd.isna(v):
        return '--'
    return f'{v:.{nd}f}'


def fp(p):
    if p < 1e-4:
        return r'$<10^{-4}$'
    return f'{p:.4f}'


def table_main(df, methods, keys=None, fname='tab_main.tex'):
    keys = keys or [m[0] for m in METRICS]
    meta = {m[0]: m for m in METRICS}
    lines = []
    for k in keys:
        _, lab, unit, nd = meta[k]
        cells = []
        for me in methods:
            ps = per_scenario(df, me)[k]
            cells.append(f'{fmt(ps.mean(), nd)} $\\pm$ {fmt(ps.std(ddof=1), nd)}')
        lines.append(f'{lab} ({unit}) & ' + ' & '.join(cells) + r' \\')
    open('results/tables/' + fname, 'w').write('\n'.join(lines))
    return lines


def table_paired(df, ref, comps, fname):
    rows = []
    X = per_scenario(df, ref)
    meta = {m[0]: m for m in METRICS}
    out = []
    for c in comps:
        Y = per_scenario(df, c)
        res = {k: paired(X[k], Y[k]) for k in ['water'] + SECONDARY}
        adj = holm([res[k]['p'] for k in SECONDARY])
        for j, k in enumerate(['water'] + SECONDARY):
            r = res[k]; nd = meta[k][3]
            padj = r['p'] if k == 'water' else adj[j - 1]
            out.append(dict(comp=c, metric=k, **r, p_holm=padj))
            rows.append(f"{LAB[c] if k=='water' else ''} & {meta[k][1]}{' (primary)' if k=='water' else ''} & "
                        f"{fmt(r['diff'], nd)} [{fmt(r['lo'], nd)}, {fmt(r['hi'], nd)}] & {fmt(r['t'], 2)} & "
                        f"{fp(r['p'])} & {fp(padj)} & {fmt(r['dz'], 2)} & {fp(r['pw'])}" + r' \\')
        rows.append(r'\hline')
    open('results/tables/' + fname, 'w').write('\n'.join(rows))
    pd.DataFrame(out).to_csv('results/tables/' + fname.replace('.tex', '.csv'), index=False)
    return out


def summary_rows(df, methods, keys, fname, with_seed=True):
    meta = {m[0]: m for m in METRICS}
    lines = []
    for me in methods:
        if me not in df.method.unique():
            continue
        ps = per_scenario(df, me)
        cells = []
        for k in keys:
            nd = meta[k][3]
            s = f'{fmt(ps[k].mean(), nd)}'
            sd = seed_sd(df, me, k)
            if with_seed and not np.isnan(sd):
                s += f' ({fmt(sd, nd)})'
            cells.append(s)
        lines.append(f'{LAB.get(me, me)} & ' + ' & '.join(cells) + r' \\')
    open('results/tables/' + fname, 'w').write('\n'.join(lines))
    return lines


# ------------------------------------------------------------------ figures
def fig_learning(methods=('pacac', 'td3lag', 'td3'), fname='figs/fig3_learning.png'):
    from sim import Spec
    sp = Spec()
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.8))
    panels = [('ret', 'Deterministic return'), ('water', 'Water (mm/14 d)'), ('egrid', 'Grid energy (kWh/14 d)'),
              ('csoil', r'Soil violation (VWC$\cdot$step)'), ('lam', r'Multipliers $\lambda_w$ (solid), $\lambda_e$ (dashed)'),
              ('q0', 'Cost-critic estimate vs realised water')]
    for me in methods:
        H = [json.load(open(f))['hist'] for f in sorted(glob.glob(f'results/models/{me}_s*_hist.json'))]
        if not H:
            continue
        ep = np.array([h['episodes'] for h in H[0]])
        for ax, (k, ttl) in zip(axs.flat, panels):
            if k in ('ret', 'water', 'egrid', 'csoil'):
                Y = np.array([[h['det'][k] for h in hh] for hh in H])
                Ys = pd.DataFrame(Y.T).rolling(5, min_periods=1).mean().values.T
                mu, sd = Ys.mean(0), Ys.std(0)
                ax.plot(ep, mu, color=COL[me], label=LAB[me]); ax.fill_between(ep, mu - sd, mu + sd, color=COL[me], alpha=0.15)
            elif k == 'lam' and me != 'td3':
                L = np.array([[h['lam'] for h in hh] for hh in H])
                ax.plot(ep, L[:, :, 0].mean(0), color=COL[me]); ax.plot(ep, L[:, :, 1].mean(0), color=COL[me], ls='--')
            elif k == 'q0' and me == 'pacac':
                Q = np.array([[h['q0'][0] * sp.W_budget for h in hh] for hh in H])
                R = np.array([[h['det']['water'] for h in hh] for hh in H])
                ax.plot(ep, Q.mean(0), color=COL[me], label='critic $Q_{\\psi_w}(o_0,\\pi(o_0))$')
                ax.plot(ep, pd.DataFrame(R.T).rolling(5, min_periods=1).mean().values.T.mean(0), color='k', ls=':', label='realised (5-ep mean)')
                ax.legend(fontsize=9)
            ax.set_title(ttl, fontsize=11); ax.set_xlabel('training episode')
    axs[0, 1].axhline(sp.W_budget, color='grey', ls='--', lw=1); axs[0, 2].axhline(sp.E_budget, color='grey', ls='--', lw=1)
    axs[1, 0].axhline(sp.d_soil, color='grey', ls='--', lw=1)
    axs[0, 0].set_ylim(bottom=200); axs[0, 1].set_ylim(0, 80); axs[0, 2].set_ylim(0, 20)
    axs[0, 0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)


def pick_scenario(df):
    r = per_scenario(df, 'thr_classic')['rain']
    return int((r - r.median()).abs().idxmin())


def fig_trajectories(scen, ctrls, fname='figs/fig4_trajectories.png'):
    """ctrls: dict label -> controller. Full 14-day trace + daily totals for one scenario."""
    from sim import Params, Spec, make_scenario, rollout
    p = Params(); sc = make_scenario(scen, p)
    logs = {lab: rollout(c, sc, p, Spec(), log=True) for lab, c in ctrls.items()}
    tday = np.arange(672) / 48
    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    ax = axs[0]
    ax.bar(tday, sc['rain'][:672], width=1 / 48, color='#6baed6', label='rainfall (mm/step)')
    ax.set_ylabel('rain (mm/30 min)'); ax.set_title(f'Test scenario {scen} (rainfall {sc["rain_total"]:.1f} mm; median-rainfall rule)')
    ax.set_xlim(0, 14)
    for lab, (m, lg) in logs.items():
        axs[1].plot(tday, lg['th'], label=lab, lw=1.2, color=COLL[lab])
        axs[2].plot(tday, np.cumsum(lg['u']), label=f'{lab}: {m["water"]:.1f} mm', color=COLL[lab])
    axs[1].axhline(p.th_min, color='k', ls='--', lw=0.8); axs[1].axhline(p.th_max, color='k', ls='--', lw=0.8)
    axs[1].axhline(p.th_star, color='grey', ls=':', lw=0.8)
    axs[1].set_ylabel(r'$\theta$ (m$^3$/m$^3$)'); axs[1].set_xlim(0, 14); axs[1].legend(fontsize=9, ncol=4)
    axs[2].set_ylabel('cumulative irrigation (mm)'); axs[2].set_xlim(0, 14); axs[2].legend(fontsize=9)
    days = np.arange(1, 15); wdt = 0.8 / len(logs)
    for j, (lab, (m, lg)) in enumerate(logs.items()):
        daily = lg['u'].reshape(14, 48).sum(1)
        axs[3].bar(days + (j - (len(logs) - 1) / 2) * wdt, daily, width=wdt, color=COLL[lab], label=f'{lab} (total {daily.sum():.1f} mm)')
    axs[3].set_xticks(days); axs[3].set_xlabel('day'); axs[3].set_ylabel('daily irrigation (mm/day)'); axs[3].legend(fontsize=9)
    for a in axs[:3]:
        a.set_xlabel('time (days)')
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    return {lab: dict(m) for lab, (m, lg) in logs.items()}, logs


COLL = {}
