# Section draft — A reciprocal dyad and a speech observation model

*(Draft for insertion after Part II. All numbers are from `results/*.csv` of the v0.7 package; 2,000 shared worlds per experiment; 95% Monte Carlo intervals for paired world-level differences. Every quantity is a simulation construct.)*

## X.1 Why a reciprocal model is needed

The single-speaker model of Part II compares two mechanisms behind public non-concession against a controlled evidence environment. In a dialogue, the evidence is the interlocutor's speech, and that speech is itself produced by an agent who is inferring the speaker. The programme's central interactional claim — that each party's response becomes the other's evidence of threat, so that protection is reciprocally sustained — cannot be stated in a single-agent model. This section specifies a minimal reciprocal dyad in which both agents carry the two Part II mechanisms, each infers the other's mechanism, and each act is rendered as a Part I speech label through an explicit observation model. The model is a simulation of stipulated mechanisms; nothing in it is fitted to people.

## X.2 Model

**World and agents.** Two agents i ∈ {A, B} disagree about a binary fact x. A prefers version 1, B version 0; each starts from a self-favouring prior (0.75) and N = 4 private records of reliability ρ = 0.75. Twelve rounds; A moves first within a round.

**Mechanisms.** Agent i carries (ω_i, c_i). c_i is the additional loss of each public CONCEDE, as in Part II. ω_i ∈ [0, 1] is evidence sensitivity on the interlocutor channel: after each exact Bayesian update on a partner act, the log-odds of i's belief about x move only ω_i times the exact increment,

  logit P_i^{new}(x=1) = logit P_i^{old}(x=1) + ω_i [ logit P̃_i(x=1) − logit P_i^{old}(x=1) ],   (1)

where P̃ is the exact posterior; inference about the partner's parameters is not attenuated. At ω_i = 1 the agent is exact; at ω_i = 0 the partner's speech leaves i's belief about the fact unchanged while i still learns who the partner is.

**Acts.** ASSERT (own version), CONCEDE (partner's version), PRESSURE, SILENCE, REASSURE. Losses: k(1 − q_own) for ASSERT, k q_own + c_i for CONCEDE, b for the three non-factual acts (k = 2, b = 1.10). REASSURE removes the partner's concession cost on the partner's next move (safe framing); it is the dyad's native intervention act.

**Level-1 inference.** Agent i maintains a joint posterior J_i(x, s_j, c_j, ω_j) over the fact, the partner's private record count and the partner's mechanism on the grids c ∈ {0, 0.8, 1.6}, ω ∈ {0.2, 0.6, 1.0}, and models the partner as a level-0 updater whose belief is logit q_j = prior_j + evidence(s_j) + ω_j M_i, where M_i is the log-odds shift i's own assertions would have induced at trusted precision λ = 0.70. The partner's act likelihood under each cell is the softmax policy above with β = 0. Both agents are level-1, so each misspecifies the other; the resulting error in each agent's estimate of the other's belief is a reported quantity.

**Policy.** One-step expected free energy, G_i(u) = ℓ_i(u) − β · IG_i(u), with IG the expected reduction in the entropy of J_i from the partner's next act after u, and P(u) ∝ exp(−G_i(u)/τ), τ = 0.20. β = 0 gives an expected-cost-only agent; β = 1.5 the active listener of the earlier dyadic line.

**Speech observation model.** Each act is rendered as one label from the Part I vocabulary {BUILD, SEEK, UNSEAL, LOCK, DRAIN, FLOOD, EDGE, VOID, SEAL, SHIFT, insufficient} by a stipulated emission matrix E_0(z | u) (Appendix), modulated with strength g ∈ [0, 1] by the speaker's mechanism: for a concession-cost speaker, ASSERT and PRESSURE shift mass toward LOCK; for an attenuated speaker, ASSERT shifts toward FLOOD and VOID; SHIFT is emitted with probability g/2 when the public position changes. At g = 0 labels carry only the information in the act code. Annotation error is then applied as an empirical confusion matrix: that of the typed engine on the 169 Tier A cases (129/169 correct) or that of the direct prompt (145/169). g is the quantity under test: it encodes the hypothesis that speech form carries mechanism information beyond the act, and the pipeline asks how much of that information survives annotation.

**Observed, inferred, assumed.**

| quantity | status |
|---|---|
| acts, public positions, labels | observed |
| x; partner's s, c, ω | inferred by each agent |
| own ω, c; k, b, τ, β, λ, ρ, N | assumed known to the agent; auxiliary for observers |
| E_0, g; confusion matrices | stipulated; empirical |
| level-0 model of the partner | assumed by each agent, false by construction |

**Observers.** An idealised observer with the full model compares the attenuation family {ω ∈ grid, c = 0} with the concession-cost family {ω = 1, c ∈ grid} for one agent from (a) both agents' acts and that agent's private records, (b) acts only (private records marginalised exactly), or (c) labels only, using a plug-in decoder that scores the agent's emissions exactly and replaces unobserved acts by their MAP decode — a lower bound on a labels-only observer.

**Validation.** With B replaced by a level-0 agent, ω = 1, a two-point cost grid and a four-act partner model, the implementation reproduces the archived v0.6 runs act-for-act with beliefs within 4 × 10⁻¹⁴; ω = 0 leaves beliefs unchanged; c → ∞ removes concession except after a partner's REASSURE; mirrored worlds with swapped move order give mirrored trajectories; a scalar re-implementation matches the vectorised policy and information gain to 10⁻⁹; the replay observer reproduces the generating act probabilities to 10⁻¹⁴.

## X.3 Dyad dynamics

**Same public outcome, opposite private states.** Among worlds in which both agents start on their own side (n = 488), the concession-cost pair and the attenuated pair are indistinguishable by late mutual non-concession (0.996 vs 0.972; honest pair 0.441) but opposite inside: the concession-cost pair has privately converged (final belief gap 0.222, hidden shift 0.266) whereas the attenuated pair has not moved (hidden shift 0.059) and ends in the largest disagreement of all pairs (0.607). This is the dyadic form of the Part II dissociation. Over all worlds the attenuated pair holds public non-concession only at 0.284 against 0.980 for the cost pair: attenuation of the interlocutor channel does not protect a public position from the speaker's own private records; a concession cost does. The epistemic term β changes the map by at most 0.03.

**Reciprocal amplification is not testable at the ceiling.** With a single concession-cost agent already at 0.998 late non-concession, the pair's mutual rate equals the independence prediction (Δ = 0.000 [0.000, 0.000]) and exceeds an additive prediction by only +0.005 [−0.002, +0.011]. Super-additivity is reliable only for the attenuated pair (+0.054 [+0.034, +0.073]).

**Who is recognised.** A level-1 agent identifies an attenuated partner (posterior mass 0.68 on ω = 0.2 against a prior of 0.33) but misreads an honest level-1 partner as attenuated (mass 0.25 on ω = 1), because a level-1 partner updates less than the level-0 model with λ = 0.70 predicts; a partner's cost c = 1.6 is recovered partially (0.55).

**Regime criteria (24 rounds).** Persistence of the mutual non-concession state exceeds 0.9 if and only if neither agent is honest (cost pair 0.983, attenuated pair 0.953, mixed pair 0.956; honest × cost 0.886; honest pair 0.525). Removing both costs in rounds 8–10 and restoring them from round 11 returns concession *acts* to control within two rounds in all 16 pairs (cost pair −0.001 [−0.006, +0.004]) but not public *positions*: divergence 0.316 vs 0.966 in control four rounds later (Δ = −0.650 [−0.691, −0.610]). Hysteresis lives in the public record, not in the policy. For next-act forecasting on 1,000 held-out worlds, both agents' last acts beat the speaker's last act by +0.031 in log-likelihood; a regime state (run of mutual non-concession ≥ 3 and divergent positions) adds nothing on top of the speaker's last act alone (−0.0004 [−0.0036, +0.0028]) and +0.026 [+0.024, +0.029] on top of both last acts, zero for the cost pair.

## X.4 Mechanism recovery from acts and from speech labels

| generator (A, partner honest) | channel | correct family | posterior on family | ω MAE (prior 0.40) | c MAE (prior 0.80) |
|---|---|---|---|---|---|
| attenuated | acts + private | 0.648 | 0.675 | 0.411 | 0.230 |
| attenuated | acts | 0.618 | 0.610 | 0.475 | 0.288 |
| concession cost | acts + private | 0.910 | 0.756 | 0.114 | 0.585 |
| concession cost | acts | 0.901 | 0.754 | 0.117 | 0.592 |

A concession cost is recovered from public acts (MAP at the true point in 0.88–0.90 of worlds); attenuation is not — its ω estimate is no better than the prior mean, and in 0.34–0.36 of worlds the attenuated speaker is read as costly. Private records add little (+0.01 to +0.05). Prediction and explanation diverge as in Part II: the best cost-family hypothesis predicts an attenuated speaker's next acts only 0.014 nats/act worse than the truth, whereas the best attenuation hypothesis predicts a costly speaker 0.474 nats/act worse. A mixed generator (0.6, 0.8) is assigned to the cost family in ~70% of worlds with ω over- and c under-estimated (0.85, 0.66).

**Labels.** At g = 0, labels recover slightly less than acts (attenuated 0.59/0.57/0.58 under no noise / engine / direct-prompt confusion vs 0.618 from acts; costly 0.89/0.90/0.90 vs 0.901), the loss coming from the non-invertible emission (LOCK ← ASSERT or PRESSURE). From g ≥ 0.25 (attenuated) and g ≥ 0.5 (costly), labels exceed both acts and acts + private records (attenuated g = 0.5: 0.948/0.940/0.948; g = 1: 0.990/0.986/0.990). The engine's confusion matrix leaves the gain essentially intact (gain ratio engine/noiseless 0.97 and 1.04 at g = 0.5) because it rarely confuses LOCK with FLOOD or DRAIN, the direction of the stipulated modulation. Hence, in this model, the entire value of speech annotation for mechanism identification is carried by g — the empirical hypothesis that speech form encodes mechanism beyond the act — and not by annotation accuracy in the range observed.

## X.5 Interventions and measurement channels

A forced act by the partner displaces the partner's assertion; the correct control is therefore a forced SILENCE at the same round, which alone lowers a costly speaker's truth-weighted belief by 0.027. Against that control, the partner's safe framing (REASSURE at round 8) raises a concession-cost speaker's probability of conceding in rounds 9–12 by +0.362 [+0.341, +0.383] without changing its belief (−0.004); a placebo framing (believed but not granted) returns concession to control (−0.008) and leaves belief unchanged (+0.000). For an attenuated speaker, framing minus control is exactly zero on every metric (identical acts in 100% of worlds), and cost removal is zero by construction. Shared strong evidence (ρ = 0.95 to both) raises the attenuated speaker's truth-weighted belief by +0.131 [+0.121, +0.141] against +0.067 [+0.059, +0.076] for the costly one — the opposite ordering from Part II, because in this model ω attenuates the interlocutor channel and not external observations; the sign of the "stronger evidence" test is therefore a property of where attenuation is located, which a human study would have to establish rather than assume.

**Simulated dialogues per group for 80% power (Welch, α = 0.05, attenuated vs costly speaker, honest partner):**

| channel | all worlds | speaker starts on own side |
|---|---|---|
| late non-concession | 17 | not reached (0.66 at 400) |
| private belief change | 25 | 25 |
| concession after cost removal | not reached (0.62 at 400) | 70 |
| concession after partner's framing | not reached | 400 |
| belief after shared evidence | not reached (0.43) | not reached (0.49) |
| listener's posterior P_B(c_A = 1.6) | 8 | 50 |

Over all worlds public channels separate the generators only because an attenuated speaker concedes where its private records favour the partner; once the speaker starts on its own side — the Part II situation — non-concession stops discriminating, and the best channels are the speaker's private belief and the interlocutor's posterior about the speaker's cost. Power was estimated by resampling the same worlds used for the contrasts. Across 38 settings of τ, k, λ, rounds and ρ, the costly-minus-attenuated contrast in private belief change is positive in 38/38 (0.045–0.174); the contrast in late non-concession is positive in 38/38 over all worlds and in 35/38 when the speaker starts on its own side (three indistinguishable from zero at τ = 0.1, k = 2); only the listener's cost posterior reverses sign (τ = 0.4, k = 1).

## X.6 What the reciprocal model does and does not establish

The model states the reciprocal claim formally and makes it testable within the simulator: whether protection in one agent raises protection in the other beyond what two independent speakers would show, and whether the interlocutor — who is inferring the speaker's mechanism — is itself a measurement instrument. It links Part I to Part II only through a stipulated observation model; the empirical question of whether real speech carries mechanism information (g > 0) is untouched. The level-0 partner model is a deliberate simplification: recursive level-k reasoning, learning of the emission model, and natural-language generation remain outside the model.
