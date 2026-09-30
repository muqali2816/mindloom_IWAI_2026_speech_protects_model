SPEECH AS CONTROL OF LEARNING OPPORTUNITIES
Exploratory dyadic bridge, 2026-09-30

START HERE
report/Mindloom_dyadic_bridge_RU_2026-09-30.pdf (8 pages): plain Russian
explanation, equations, observed/inferred/assumed distinctions, results,
related-work limits and English insertion text.
report/Proposed_bridge_sections_EN.txt: editable English insertion text.

PURPOSE
Connect the speech annotation proposal to an explicit reciprocal process:
interactional forms invite or restrict joint verification, affecting both
agents' subsequent evidence and belief updating. Test whether noisy form
annotations add next-evidence prediction beyond observed coarse actions and
past evidence. This is an exploratory computational proposal, not a human
validation or a claim to have established field-wide novelty.

REPRODUCTION
Python 3 + numpy + pandas:
  python run_experiments.py
  python validate_independent.py
Optional report rebuild: matplotlib + reportlab + DejaVu Sans fonts:
  python build_report.py
Seeds, samples, all conditions and parameters are in results/design.json.
All tested outcomes are retained. No tuning after examining pilot results.

MODEL
Two agents infer one static binary fact, with priors 0.75 and 0.25. Twelve
exchanges contain 24 individual decisions. Each chooses one of five coarse
acts and one of two forms (invite/restrict). Joint forms control the chance of
fresh shared evidence; both agents update on each new shared signal once.
Agents forecast the other by holding its previous public move fixed. This is
bounded reciprocal influence, not recursive theory of mind or equilibrium.
Gamma prices expected public contradiction and is distinct from concession
cost c, evidence sensitivity omega and restriction cost delta. All costs are
dimensionless. No physiological load or higher-order self-worth is modelled.

MEASUREMENT
The archived 169-case engine predictions are copied unchanged in source/.
The original vote-based scorer is used to extract the confusion matrix.
Only two gold rows, LOCK (15) and SEEK (20), enter this prototype. Their raw
output supports are disjoint. Applying them literally to a two-form world
would make forms perfectly identifiable, an unrealistic advantage. The
primary demonstration therefore uses 0.5 pseudocount per output and 25%
state-independent noise, with additional noise sensitivity. These are stated
transport assumptions, NOT measured accuracy on new dyadic data. Kappa is
not converted to accuracy. No LLM verbalizer or live engine is run.
The 11 historical output categories are retained as possible noisy detector
outputs; they are not 11 latent regimes. SHIFT and SEAL remain different
output types. No temporal phase or sustained regime is inferred here.

OBSERVER
Exact filtering over hidden (fact, form pair), conditional on four discrete
known parameter hypotheses. All comparisons observe both agents' coarse acts
and already delivered evidence. Labels are additional observations. No future
evidence is available when the next-evidence probability is predicted.
Prediction log loss is averaged over targets e_5 through e_12, i.e. forecasts
made after exchanges 4 through 11. New worlds (seed 202609303) are distinct
from descriptive/intervention worlds (202609302). There is no fitted training
set because the observer is an analytical known-model filter.
This task is NOT directly comparable with original single-agent Table 5.

MAIN FINDINGS IN THIS SPECIFICATION
With omega=1 but gamma=2 for both agents, evidence count falls from 2.701 to
1.8205 and mean final truth-weighted belief from 0.7076 to 0.6580.
Noisy labels improve next-evidence log loss by 6.9-8.9% across four specified
generators. Gains vanish with conditional-action-only labels, completely
uninformative labels, and when forms do not affect access. Most of the learning
loss survives removal of expected information gain. Thus this is not evidence
of unique explanatory superiority for active inference over utility models.

VALIDATION
Independent scalar policy probabilities and exhaustive short latent-path
enumeration check the vectorized policy and exact observer. Null channels and
coupling ablation pass within 1e-12. See results/validation.json and
results/independent_validation.json. All CIs are Monte Carlo mean +/- 1.96 SE
across shared simulated worlds, not intervals for human populations.

IMPORTANT LIMITS
The impact of an interactional form on access is stipulated, not estimated.
The full annotation ontology has not been validated or shown better than a
simple invitation/restriction feature. The empirical noise matrix is small
and context transport is assumed; row uncertainty is not fully integrated.
The generator does not infer hidden motives, generate natural-language
utterances, establish stable attractors or test allostatic load. A richer
semantic and empirical bridge remains necessary.

RELATED WORK
Friston & Frith (2015), A Duet for One, DOI 10.1016/j.concog.2014.12.003.
Medrano & Sajid (2024), A Broken Duet, DOI 10.3390/e26090731.
Tison & Poirier (2021), ecological communication, DOI 10.3389/fpsyg.2021.708780.
Svet & Perera Molligoda Arachchige (2026), Defense as Precision Management,
DOI 10.2139/ssrn.7245984 (abstract checked; full text not obtained here).
Dyads, evidence-channel restriction and interactional affordances are not
claimed as new. The proposed contribution concerns the explicit measurement
bridge and tests of incremental predictive value. Priority is not established
by this bounded literature check.

The original manuscript and original simulation package have not been changed.
