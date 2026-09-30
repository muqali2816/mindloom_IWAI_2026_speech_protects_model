# Reciprocal dyad v0.7 — two level-1 active-inference agents with (omega, c) mechanisms and a speech-label layer

    PYTHONPATH=. python validate.py --v06 ../dd/defensive_dialogue_v06   # 10 checks incl. bitwise regression to v0.6
    PYTHONPATH=. python E1_phase_map.py      # etc. — experiment scripts write results/ and figures/

Files: MODEL_SPEC.md (formal model), protocol.json (grids, seeds, predictions P1–P5 written before runs),
core.py (worlds, Level1/Level0 agents, dialogue, speech-label emission + noise, metrics),
observer.py (mechanism-recovery observers: acts+private, acts, labels[plug-in decoder]),
data/*_confusion_rownorm_tierA.csv (empirical confusion matrices of the typed engine 129/169 and the direct prompt 145/169,
recomputed 30.09.2026 from engine_md3_combined.json / eval_naive_baseline_en_1782919205.json, label-majority of 3 runs).

Conventions: A prefers version 1, B version 0; A moves first within a round (cond['order']='BA' swaps);
cond = {'A': {'omega','c','beta'}, 'B': {...}, 'B_level': 1|0, 'intervention': {...}}.
Interventions: {'force_reassure': {'round': 7, 'agent': 'B'}}, {'placebo': True}, {'remove_c_from': 7, 'remove_c_for': ['A']},
{'restore_c_from': 10, ...}, {'external_obs_round': 7, 'external_rho': 0.95}.
Attenuation rule: core.ATTENUATION = 'shrink' (default; see MODEL_SPEC §4 note), alternatives 'mixture', 'power'.
