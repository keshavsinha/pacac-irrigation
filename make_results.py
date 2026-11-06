"""Produce every table, figure and number used in the revised manuscript."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import analysis as A
from analysis import per_scenario, paired, holm, LAB, COL
import matplotlib.pyplot as plt
from sim import Params, Spec, make_scenario, rollout, N_STEPS
from evaluate import learned, baseline

os.makedirs('results/tables', exist_ok=True); os.makedirs('figs', exist_ok=True)
N = {}
df = A.load('nominal')
MAIN = ['thr_classic', 'thr_tuned', 'mpc', 'td3', 'td3lag', 'pacac']

# ---------------- Table: main summary (mean +- SD over 20 scenarios; learned = mean over seeds) -------------
keys = ['water', 'egrid', 'epump', 'csoil', 'viol_steps', 'mae_theta', 'crop_proxy', 'rel_transp',
        'P_mean', 'P_peak', 'hi_steps', 'hi_irr', 'sup', 'sup_mag', 'ret']
A.table_main(df, MAIN, keys, 'tab_main.tex')
summ = {}
for me in MAIN:
    ps = per_scenario(df, me)
    summ[me] = {k: dict(mean=float(ps[k].mean()), sd=float(ps[k].std(ddof=1)), seed_sd=float(A.seed_sd(df, me, k)) if me in ('td3', 'td3lag', 'pacac') else None) for k in keys + ['w_exceed', 'e_exceed', 's_exceed', 'rain']}
N['summary'] = summ
# exceedance rows (fraction of realizations; learned: pooled over seeds)
ex = []
for me in MAIN:
    d = df[df.method == me]
    ex.append(f"{LAB[me]} & {d.w_exceed.mean()*100:.0f} & {d.e_exceed.mean()*100:.0f} & {d.s_exceed.mean()*100:.0f} \\\\")
open('results/tables/tab_exceed.tex', 'w').write('\n'.join(ex))
# seed-level variability of learned methods
seedrows = []
for me in ['td3', 'td3lag', 'pacac']:
    d = df[df.method == me].groupby('train_seed')[['water', 'egrid', 'ret', 'P_mean']].mean()
    seedrows.append(f"{LAB[me]} & " + ' & '.join(f"{d[k].mean():.2f} $\\pm$ {d[k].std(ddof=1):.2f} [{d[k].min():.2f}, {d[k].max():.2f}]" for k in ['water', 'egrid', 'P_mean', 'ret']) + r' \\')
open('results/tables/tab_seeds.tex', 'w').write('\n'.join(seedrows))

# ---------------- paired tests ----------------
pt = A.table_paired(df, 'pacac', ['thr_classic', 'thr_tuned', 'mpc', 'td3', 'td3lag'], 'tab_paired.tex')
N['paired'] = pt
# normality of paired differences + Wilcoxon reported in table

# ---------------- ablations ----------------
ABL = ['pacac', 'pacac_nopstate', 'pacac_noprew', 'pacac_nopest', 'pacac_noaug', 'pacac_cgrad', 'pacac_penalty', 'td3lag', 'td3']
akeys = ['water', 'egrid', 'csoil', 'P_mean', 'hi_steps', 'hi_irr', 'crop_proxy', 'ret']
A.summary_rows(df, ABL, akeys, 'tab_ablation.tex')
abl = {}
for me in ABL:
    if me in df.method.unique():
        ps = per_scenario(df, me)
        d = df[df.method == me]
        abl[me] = {k: float(ps[k].mean()) for k in akeys}
        abl[me].update(w_exceed=float(d.w_exceed.mean()), e_exceed=float(d.e_exceed.mean()), n_seeds=int(d.train_seed.nunique()))
N['ablation'] = abl
# pest-aware timing: disease index of PAC-AC vs no-pest variant, paired over scenarios
if 'pacac_nopest' in df.method.unique():
    X, Y = per_scenario(df, 'pacac'), per_scenario(df, 'pacac_nopest')
    N['pest_vs_nopest'] = {k: paired(X[k], Y[k]) for k in ['P_mean', 'hi_steps', 'water', 'egrid', 'ret']}
    dfc = A.load('couple0')
    if 'pacac' in dfc.method.unique() and 'pacac_nopest' in dfc.method.unique():
        Xc, Yc = per_scenario(dfc, 'pacac'), per_scenario(dfc, 'pacac_nopest')
        N['pest_vs_nopest_couple0'] = {k: paired(Xc[k], Yc[k]) for k in ['P_mean', 'water', 'ret']}

# ---------------- sensitivity ----------------
SENS = ['pacac', 'sens_a4x0.5', 'sens_a4x2', 'sens_Wx0.85', 'sens_Wx1.15']
A.summary_rows(df, SENS, ['water', 'egrid', 'P_mean', 'hi_steps', 'crop_proxy', 'ret'], 'tab_sens.tex', with_seed=False)
N['sens'] = {me: {k: float(per_scenario(df, me)[k].mean()) for k in ['water', 'egrid', 'P_mean', 'ret']} for me in SENS if me in df.method.unique()}

# ---------------- robustness ----------------
allc = pd.concat([pd.read_csv(f) for f in glob.glob('results/eval/*.csv')])
CONDS = ['nominal', 'noisy_kf', 'noisy_raw', 'bias+0.02', 'bias-0.02', 'dropout30', 'pest_misspec+3C', 'couple0', 'couple0.5', 'couple2',
         'soil_Zr300', 'soil_clayloam', 'rain_x0.5', 'rain_x1.5', 'hot+2C', 'Topt+3C_true']
CL = {'nominal': 'Nominal', 'noisy_kf': 'Sensor noise + KF', 'noisy_raw': 'Sensor noise, no filter', 'bias+0.02': r'VWC bias $+0.02$',
      'bias-0.02': r'VWC bias $-0.02$', 'dropout30': '30\\% VWC dropout + KF', 'pest_misspec+3C': r'Edge disease model $T_o{+}3^\circ$C',
      'couple0': 'No irrigation--canopy coupling (drip-like)', 'couple0.5': r'Coupling $\times0.5$', 'couple2': r'Coupling $\times2$',
      'soil_Zr300': r'$Z_r=300$ mm', 'soil_clayloam': 'Clay-loam soil', 'rain_x0.5': r'Rainfall $\times0.5$', 'rain_x1.5': r'Rainfall $\times1.5$',
      'hot+2C': r'Temperature $+2^\circ$C', 'Topt+3C_true': r'True pathogen $T_o{+}3^\circ$C'}
rob_rows = []; rob = {}
for cnd in CONDS:
    cells = []
    for me in ['thr_tuned', 'mpc', 'td3', 'pacac']:
        d = allc[(allc.cond == cnd) & (allc.method == me)]
        if len(d) == 0:
            cells += ['--', '--', '--']; continue
        ps = d.groupby('scenario').mean(numeric_only=True)
        cells += [f"{ps.water.mean():.1f}", f"{ps.P_mean.mean():.3f}", f"{ps.ret.mean():.0f}"]
        rob[(cnd, me)] = dict(water=float(ps.water.mean()), P=float(ps.P_mean.mean()), ret=float(ps.ret.mean()), egrid=float(ps.egrid.mean()),
                              csoil=float(ps.csoil.mean()), sup=float(ps.sup.mean()))
    rob_rows.append(f"{CL[cnd]} & " + ' & '.join(cells) + r' \\')
open('results/tables/tab_robust.tex', 'w').write('\n'.join(rob_rows))
N['robust'] = {f'{k[0]}|{k[1]}': v for k, v in rob.items()}

# ---------------- layered soil ----------------
if os.path.exists('results/eval_layered.csv'):
    L = pd.read_csv('results/eval_layered.csv')
    rows = []
    for me in ['thr_classic', 'thr_tuned', 'td3', 'pacac']:
        cells = []
        for pr in range(4):
            d = L[(L.method == me) & (L.probe == pr)].groupby('scenario').mean(numeric_only=True)
            cells.append(f"{d.water.mean():.1f} / {d.rel_transp.mean():.2f} / {d.top_sat_steps.mean():.0f}")
        rows.append(f"{LAB[me]} & " + ' & '.join(cells) + r' \\')
    open('results/tables/tab_layered.tex', 'w').write('\n'.join(rows))
    N['layered'] = L.groupby(['method', 'probe'])[['water', 'rel_transp', 'mae_theta', 'P_mean', 'top_sat_steps', 'deep_def_steps', 'csoil']].mean().reset_index().to_dict('records')

# ---------------- figures ----------------
A.fig_learning(('pacac', 'td3lag', 'td3'))
scen = A.pick_scenario(df); N['trace_scenario'] = scen
A.COLL.update({'Threshold (classical)': COL['thr_classic'], 'Threshold (tuned)': COL['thr_tuned'], 'MPC': COL['mpc'], 'PAC-AC (seed 0)': COL['pacac']})
ctrls = {'Threshold (classical)': baseline('thr_classic'), 'Threshold (tuned)': baseline('thr_tuned'), 'MPC': baseline('mpc'),
         'PAC-AC (seed 0)': learned('pacac', 0)[0]}
tr, logs = A.fig_trajectories(scen, ctrls)
N['trace_totals'] = {k: dict(water=v['water'], egrid=float(v['egrid']), P_mean=float(v['P_mean'])) for k, v in tr.items()}

# disease-index distribution (all half-hourly values, all scenarios) + hour-of-day irrigation profile
p = Params()
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
dist = {}
for me in ['thr_classic', 'thr_tuned', 'mpc', 'td3', 'pacac']:
    Ps, Uh, daily = [], np.zeros(48), []
    seeds = [None] if me in ('thr_classic', 'thr_tuned', 'mpc') else [0, 1, 2, 3, 4]
    scen_list = range(1, 21)
    for sd in seeds:
        c = baseline(me) if sd is None else learned(me, sd)[0]
        for s in scen_list:
            m, lg = rollout(c, make_scenario(s, p), p, Spec(), log=True)
            Ps.append(lg['P']); Uh += lg['u'].reshape(14, 48).sum(0); daily.append(lg['P'].reshape(14, 48).mean(1))
    Ps = np.concatenate(Ps)
    dist[me] = dict(q50=float(np.median(Ps)), q95=float(np.quantile(Ps, 0.95)), q99=float(np.quantile(Ps, 0.99)), max=float(Ps.max()),
                    frac_gt05=float((Ps > 0.5).mean()), n=int(len(Ps)))
    axs[0].hist(Ps, bins=60, range=(0, 0.8), histtype='step', density=True, color=COL[me], label=LAB[me], lw=1.5)
    D = np.array(daily); mu = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(D))
    axs[1].plot(np.arange(1, 15), mu, color=COL[me], label=LAB[me]); axs[1].fill_between(np.arange(1, 15), mu - 1.96 * se, mu + 1.96 * se, color=COL[me], alpha=0.15)
    axs[2].plot(np.arange(48) / 2 + 0.25, Uh / (len(seeds) * len(scen_list) * 14), color=COL[me], label=LAB[me])
axs[0].axvline(0.5, color='k', ls='--', lw=1); axs[0].set_xlabel('half-hourly disease index $P_t$'); axs[0].set_ylabel('density'); axs[0].set_title('(a) distribution of $P_t$ (all steps, all runs)')
axs[1].set_xlabel('day'); axs[1].set_ylabel('daily mean $P_t$'); axs[1].set_title('(b) daily mean $P_t$ (mean, 95% CI over runs)')
axs[2].set_xlabel('hour of day'); axs[2].set_ylabel('mean irrigation (mm per 30 min)'); axs[2].set_title('(c) irrigation timing'); axs[2].axvspan(8, 17, color='gold', alpha=0.1)
axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig5_disease.png', dpi=200); plt.close(fig)
N['P_dist'] = dist

# water vs disease index across methods/ablations (water-matched comparison)
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for me in ['thr_classic', 'thr_tuned', 'mpc', 'td3', 'td3lag', 'pacac', 'pacac_nopstate', 'pacac_noprew', 'pacac_nopest', 'pacac_noaug', 'pacac_cgrad', 'pacac_penalty']:
    d = df[df.method == me]
    if len(d) == 0:
        continue
    g = d.groupby('train_seed')[['water', 'P_mean']].mean()
    ax.scatter(g.water, g.P_mean, s=30, label=LAB[me], marker='o' if me in COL else 's', color=COL.get(me))
ax.axvline(Spec().W_budget, color='grey', ls='--', lw=1)
ax.set_xlabel('applied water (mm per 14 d)'); ax.set_ylabel('mean disease index'); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig('figs/fig6_water_vs_disease.png', dpi=200); plt.close(fig)

# learning diagnostics numbers
H = {me: [json.load(open(f)) for f in sorted(glob.glob(f'results/models/{me}_s*_hist.json'))] for me in ['pacac', 'td3', 'td3lag']}
N['final_lambda'] = {me: np.mean([h['hist'][-1]['lam'] for h in H[me]], 0).tolist() for me in H}
N['selected_round'] = {me: [h['cfg']['selected_round'] for h in H[me]] for me in H}
N['train_time_min'] = {me: float(np.mean([h['hist'][-1]['time'] for h in H[me]]) / 60) for me in H}
N['updates'] = {me: int(H[me][0]['hist'][-1]['updates']) for me in H}
json.dump(N, open('results/numbers.json', 'w'), indent=1, default=float)
print('done')
