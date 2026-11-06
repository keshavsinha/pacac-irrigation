"""Named training configurations (all other settings = agents.DEFAULT_CFG).
PAC-AC (final): Lagrangian-penalised reward with multipliers from Monte-Carlo undiscounted
episode constraint returns, time/budget-augmented observation, disease-aware state and reward."""
CONFIGS = {
    'td3':            dict(cost_mode='none', obs='env', lag_in_reward=False),
    'td3lag':         dict(cost_mode='lag_disc', obs='env', lag_in_reward=False),
    'pacac':          dict(cost_mode='pacac', obs='env+aug'),
    'pacac_cgrad':    dict(cost_mode='pacac', obs='env+aug', lag_in_reward=False),   # cost-critic actor gradient
    'pacac_noaug':    dict(cost_mode='pacac', obs='env'),
    'pacac_nopstate': dict(cost_mode='pacac', no_pest_state=True),
    'pacac_noprew':   dict(cost_mode='pacac', a4=0.0),
    'pacac_nopest':   dict(cost_mode='pacac', no_pest_state=True, a4=0.0),
    'pacac_penalty':  dict(cost_mode='penalty', obs='env+aug', penalty_kappa=[25.0, 10.0, 1.0]),
    'sens_a4x0.5':    dict(cost_mode='pacac', a4=0.5),
    'sens_a4x2':      dict(cost_mode='pacac', a4=2.0),
    'sens_Wx0.85':    dict(cost_mode='pacac', W_budget=38.0 * 0.85),
    'sens_Wx1.15':    dict(cost_mode='pacac', W_budget=38.0 * 1.15),
}
