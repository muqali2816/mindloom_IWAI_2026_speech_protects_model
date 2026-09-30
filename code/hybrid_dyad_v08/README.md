# Hybrid dyad v0.8 — act x form policies, joint verification channel, level-1 inference about the partner

    PYTHONPATH=. python validate.py --mbridge ../user_bridge --v07 ../reciprocal_dyad   # 10 checks incl. two bitwise regressions

core.py: make_worlds(n, seed, N) -> worlds; dialogue(worlds, cond) -> dict (acts, forms, ev, rates, L, pos, est, prob, relief, pC/pW partner);
metrics(d) -> DataFrame per world; paired(dfa, dfb, label) -> paired MC contrasts; emit_labels(d, seed) -> (form labels, null labels, rows);
channel_mean(eta) -> observer's (2,11) rows (row 0 restrict=LOCK, row 1 invite=SEEK).
cond = {'A': {'omega','c','gamma'}, 'B': {...}, 'beta': 1.5, 'order': 'sim', 'intervention': {...}}; agent types in CFG['agent_types'].
Interventions: bypass_from, remove_gamma_from / remove_c_from (+ remove_for ['A']), force_open {'agent','from'}, force_reassure {'round','agent'}, placebo, external_obs_round (+external_rho).
observer.py: recover(d, cond, channel, spec_B, labels=, C=, null_labels=) -> family_post (attenuation/cost/access), parameter means, next_sig (n,R-1), next_act (n,R-1,nA);
channels 'acts' | 'labels' | 'oracle' | 'null'; ~45 s per channel per 2000 worlds (8 hypotheses).
All quantities are simulation constructs; no human data.
