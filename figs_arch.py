"""Redrawn architecture (Fig. 1) and state-conditioning (Fig. 2) diagrams, larger text.
Solid boxes = components implemented in the simulation study; dashed = deployment design only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
IMPL = '#1f4e79'
DES = '#7f7f7f'


def box(ax, x, y, w, h, title, lines=(), impl=True, fc='white', tsize=12.5, lsize=11):
    ls = '-' if impl else (0, (5, 3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=IMPL if impl else DES, lw=1.8, ls=ls))
    ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=tsize, weight='bold')
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - 0.36 * i, l, ha='center', va='top', fontsize=lsize)


def arrow(ax, x1, y1, x2, y2, text=None, dashed=False, tx=0, ty=0.12, fs=11):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='k', ls='--' if dashed else '-',
                                mutation_scale=14))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha='center', va='bottom', fontsize=fs)


def tier(ax, x, y, w, h, label, fc, right=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc=fc, ec='#444444', lw=1.2))
    if right:
        ax.text(x + w - 0.6, y + h - 0.12, label, ha='right', va='top', fontsize=14, weight='bold')
    else:
        ax.text(x + 0.15, y + h - 0.12, label, ha='left', va='top', fontsize=14, weight='bold')


# ---------------- Fig. 1 ----------------
fig, ax = plt.subplots(figsize=(14, 8.7))
ax.set_xlim(0, 14); ax.set_ylim(0, 8.7); ax.axis('off')
tier(ax, 0.1, 5.35, 13.8, 2.75, 'Tier 2 - Edge gateway (inference every 30 min)', '#eaf2fb')
tier(ax, 0.1, 2.55, 13.8, 2.6, 'Tier 1 - Field node (STM32-class MCU)', '#eef7ea')
tier(ax, 0.1, 0.1, 13.8, 2.25, 'Tier 3 - Cloud (offline training, periodic update)', '#f3eefa', right=True)
box(ax, 0.4, 5.55, 3.0, 1.95, 'State estimation', ['KF on soil moisture (sim.)', 'EWMA on T, RH', 'UKF multi-rate (design)'])
box(ax, 3.8, 5.55, 3.0, 1.95, 'Disease-index model', ['Magarey temperature', 'response x leaf wetness', '-> index P_t in [0,1]'])
box(ax, 7.2, 5.55, 3.1, 1.95, 'Observation composer', ['o_t = [env. features,', 'time-of-day, tau_t,', 'remaining budgets]'])
box(ax, 10.7, 5.55, 3.0, 1.95, 'PAC-AC actor only', ['14-64-64-1 MLP', 'int8: 5.1 KB weights', '(critics stay in cloud)'])
box(ax, 0.4, 2.75, 3.6, 1.95, 'Sensor suite (11 ch.)', ['VWC, soil T, EC, air T, RH,', 'leaf wetness, PAR, NDVI,', 'rain, wind, pest evidence'])
box(ax, 4.4, 2.75, 3.6, 1.95, 'Pre-processing', ['outlier rejection,', 'sync to 30-min grid,', 'logging + watchdog'], impl=False)
box(ax, 8.4, 2.75, 3.0, 1.95, 'Safety supervisor', ['S1 actuator bounds', 'S2 saturation guard', 'S3 deficit guard'])
box(ax, 11.7, 2.75, 1.65, 1.95, 'Pump/valve', ['micro-', 'sprinkler,', 'PV + grid'])
box(ax, 0.4, 0.3, 3.6, 1.55, 'Transition log store', ['(o_t, u_t, r_t, c_t, o_t+1)', 'uploaded ~64 B/h'], impl=False)
box(ax, 4.6, 0.3, 4.6, 1.55, 'PAC-AC trainer (simulation)', ['twin reward critics, 3 cost critics,', 'Monte-Carlo dual update of lambda'])
box(ax, 9.8, 0.3, 3.9, 1.55, 'Update package', ['actor weights only', '~5.2 KB per weekly update'], impl=False)
arrow(ax, 3.4, 6.5, 3.8, 6.5); arrow(ax, 6.8, 6.5, 7.2, 6.5); arrow(ax, 10.3, 6.5, 10.7, 6.5)
arrow(ax, 2.2, 4.7, 2.2, 5.55, 'z_t', tx=0.25, ty=0.15)
arrow(ax, 4.0, 3.7, 4.4, 3.7)
arrow(ax, 11.5, 5.55, 10.4, 4.7, 'u_cmd', tx=0.6, ty=-0.05)
arrow(ax, 11.4, 3.7, 11.7, 3.7, 'u_t', ty=0.1)
arrow(ax, 5.0, 2.75, 3.0, 1.85, dashed=True)
arrow(ax, 4.0, 1.15, 4.6, 1.15, dashed=True); arrow(ax, 9.2, 1.15, 9.8, 1.15, dashed=True)
arrow(ax, 13.55, 1.85, 13.55, 5.55, dashed=True)
ax.text(13.6, 4.95, 'deploy', ha='left', fontsize=10.5, rotation=90)
ax.plot([7.4, 8.0], [8.45, 8.45], color=IMPL, lw=1.8); ax.text(8.1, 8.45, 'used in simulation study', va='center', fontsize=10.5)
ax.plot([10.6, 11.2], [8.45, 8.45], color=DES, lw=1.8, ls=(0, (5, 3))); ax.text(11.3, 8.45, 'deployment design only', va='center', fontsize=10.5)
fig.tight_layout(); fig.savefig('figs/fig1_architecture.png', dpi=220); plt.close(fig)

# ---------------- Fig. 2 ----------------
fig, ax = plt.subplots(figsize=(14, 4.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis('off')
box(ax, 0.2, 1.1, 2.6, 2.6, 'Raw sensing', ['VWC (1 probe,', '15 cm depth),', 'T, RH, leaf wet.,', 'rain, NDVI (daily)'], lsize=10.5)
box(ax, 3.2, 1.1, 2.6, 2.6, 'Tier 1', ['outlier rejection', 'sync to dt=30 min', 'gap fill <= 3 h', '-> z_t'], impl=False, lsize=10.5)
box(ax, 6.2, 1.1, 2.6, 2.6, 'Tier 2', ['KF on theta with', 'water-balance', 'prediction;', 'EWMA on T, RH'], lsize=10.5)
box(ax, 9.2, 1.1, 2.1, 2.6, 'Tier 3', ['disease index', 'P_t from est.', 'T, leaf wet.,', 'NDVI'], lsize=10.5)
box(ax, 11.7, 1.1, 2.1, 2.6, 'Observation', ['o_t (14-dim)', '-> actor', '-> supervisor', '(theta_hat)'], lsize=10.5)
for x in [2.8, 5.8, 8.8, 11.3]:
    arrow(ax, x, 2.4, x + 0.4, 2.4)
ax.text(7.0, 0.5, 'Main experiments use true simulator states; the noisy-sensor + estimator path is evaluated in the robustness tests (Section 6.7).',
        ha='center', fontsize=11, style='italic')
fig.tight_layout(); fig.savefig('figs/fig2_state_conditioning.png', dpi=220); plt.close(fig)
print('ok')
