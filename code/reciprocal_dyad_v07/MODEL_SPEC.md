# Reciprocal dyad v0.7 — formal specification

Status: specification written before implementation and before any run. Extends the
defensive-dialogue kernel v0.4–v0.6 (Mindloom, September 2026). No human data, no fitted
human parameters, no NLP. All quantities are simulation constructs.

## 1. World

- Hidden fact x ∈ {0, 1}: which of two versions is true. Balanced across worlds.
- Two agents i ∈ {A, B}. A prefers version 1, B prefers version 0 (`PREF = [1, 0]`).
- Each agent holds N = 4 private records s_i ~ Binomial(N, ρ) with ρ = 0.75 matching x.
- Self-favouring prior 0.75 on the preferred version. Initial log-odds for version 1:
  L_i(0) = ±logit(0.75) + (2 s_i − N)·llr(ρ), llr(ρ) = ln(ρ/(1−ρ)).
- R = 12 rounds; within a round A acts, then B acts (order fixed, as in v0.4–v0.6).

## 2. Agent parameters (mechanisms)

Each agent i carries two mechanism parameters, the same two compared in Part II of the paper:

| symbol | range | meaning | Part II analogue |
|---|---|---|---|
| ω_i | [0, 1] | evidence sensitivity on the **interlocutor channel**: after each exact joint update on a partner act, the log-odds of i's belief about x move only ω_i times the exact increment (partner-parameter inference is untouched). ω_i = 1 exact Bayes; ω_i = 0 partner's speech leaves i's belief about x unchanged | likelihood attenuation ω |
| c_i | ≥ 0 | additional loss of each public CONCEDE | concession cost c |

Private records are not attenuated: ω concerns what the dialogue can change.

## 3. Acts (five, identical menu for both agents)

| code | act | factual content | effect on partner |
|---|---|---|---|
| 0 | ASSERT (say_own) | asserts own preferred version | evidence for partner; sets public position |
| 1 | CONCEDE | asserts partner's version | evidence for partner; sets public position |
| 2 | PRESSURE | none | none (kept for continuity with v0.4–v0.6 and Part II) |
| 3 | SILENCE | none | none |
| 4 | REASSURE (safe framing) | none | partner's CONCEDE cost is 0 on partner's **next** move |

Loss of act u for agent i with q_own = P_i(own preferred version), k = 2, b = 1.10:

ℓ(ASSERT) = k(1 − q_own); ℓ(CONCEDE) = k·q_own + c_i·(1 − relief_i); ℓ(PRESSURE) = ℓ(SILENCE) = ℓ(REASSURE) = b.

relief_i = 1 iff the partner chose REASSURE on its immediately preceding move.

## 4. Each agent's generative model of the partner (level-1 reasoning)

Agent i represents partner j as a **level-0 updater** with unknown (s_j, c_j, ω_j):

- partner belief in version 1: logit q_j = ±logit(0.75) + (2 s_j − N)·llr(ρ) + ω_j · M_i,
  where M_i = Σ over i's past factual acts of ±llr(λ), λ = 0.70 (trusted precision), i.e. the
  log-odds shift i believes its own assertions induced in j;
- partner acts by the softmax policy of §3 with β = 0 (no epistemic term) and cost c_j
  (relieved if i reassured on the previous move);
- i's joint posterior J_i(x, s_j, c_j, ω_j) starts as P_i(x) · Binomial(s_j | x) · U(c_j) · U(ω_j)
  on the grids c ∈ {0, 0.8, 1.6}, ω ∈ {0.2, 0.6, 1.0} (3 × 3; v0.6 used c ∈ {0, 1.6}, ω ≡ 1).

Update after observing partner act a_j (two steps):

1. exact: J̃ ∝ J_i · P(a_j | x, s_j, c_j, ω_j, M_i, pos_i, relief_j);
2. attenuation of the fact channel: logit P_new(x=1) = logit P_old(x=1) + ω_i·[logit P̃(x=1) − logit P_old(x=1)];
   J_i ← J̃ with its x-marginal rescaled to P_new. At ω_i = 1 the step is exact Bayes.

*Implementation note (30.09.2026, before any experiment run).* The specification first considered
act-likelihood tempering P^{ω} (as in Part II). Because the level-0 act likelihoods are nearly
deterministic (τ = 0.20), tempering leaves per-act likelihood ratios large and repeated partner
assertions accumulate: at ω = 0.2 the attenuated agent's belief change equalled the honest agent's
(0.15 vs 0.13). A mixture with uniform noise (ω·P + (1−ω)/5) bounds each act's evidence but
repetition still converges (0.125 at ω = 0.2). The marginal-shrink rule above is the only one of
the three under which ω caps the total influence of the dialogue on the fact (belief change 0.034
at ω = 0.2, 0 at ω = 0) and it coincides with the level-0 partner model, where ω_j multiplies the
induced shift M_i. Both alternatives remain selectable in `core.ATTENUATION` for sensitivity.

i's belief about x is the marginal L_i = logit Σ_{s,c,ω} J_i(x = 1, ·). Both agents are level-1;
each therefore misspecifies the other (the other is level-1, not level-0). The size of that
misspecification is a reported quantity ("transparency error": |estimated − actual partner belief|).

## 5. Action selection (one-step expected free energy)

For each candidate act u, expected information gain about the joint latent state
(x, s_j, c_j, ω_j) from the partner's **next** act:

IG_i(u) = H[J_i] − Σ_{a_j} P(a_j | u) · H[J_i | u, a_j],

with the predictive P(a_j | u) computed under i's model of j after i's act u (ASSERT/CONCEDE
shift M_i and pos_i; REASSURE sets relief_j). Policy:

P(u) ∝ exp{ [−ℓ_i(u) + β · IG_i(u)] / τ }, τ = 0.20, β ∈ {0, 1.5}.

β = 0 recovers an expected-cost-only agent; β = 1.5 is the v0.6 active listener.

## 6. Limiting cases (validation targets)

| condition | must reduce to |
|---|---|
| B replaced by level-0 with true cost, ω_A = 1, point prior on c, ω-grid {1}, 4 acts, β = 0 | v0.6 `REG_*` runs: acts identical, beliefs ≤ 1e-10 |
| same but β = 1.5, 5 acts | v0.6 `C1.6__latent__beta1.5` (acts identical, beliefs ≤ 1e-10) |
| ω_i → 0 for both | neither agent's belief moves from L_i(0) |
| c_i → ∞ | CONCEDE probability → 0 for agent i |
| A ↔ B role swap with mirrored worlds | mirrored statistics |

## 7. Speech-label observation layer (link to Part I)

Each act is rendered as one label z_t from the Part I vocabulary
Z = {BUILD, SEEK, UNSEAL, LOCK, DRAIN, FLOOD, EDGE, VOID, SEAL, SHIFT, INSUFF}.

**Base emission E0(z | u)** (stipulated; g = 0):

| act | BUILD | SEEK | UNSEAL | LOCK | DRAIN | FLOOD | EDGE | VOID | SEAL | INSUFF |
|---|---|---|---|---|---|---|---|---|---|---|
| ASSERT | 0 | 0 | 0 | .30 | 0 | .50 | .10 | 0 | 0 | .10 |
| CONCEDE | .60 | 0 | .30 | 0 | 0 | 0 | 0 | 0 | 0 | .10 |
| PRESSURE | 0 | 0 | 0 | .50 | .20 | .20 | 0 | 0 | 0 | .10 |
| SILENCE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .50 | .40 | .10 |
| REASSURE | .30 | .50 | .10 | 0 | 0 | 0 | 0 | 0 | 0 | .10 |

**Mechanism modulation, strength g ∈ [0, 1]** (the quantity under test):

- concession-cost agent (c_i > 0): for ASSERT and PRESSURE, move mass g·(c_i / c_max) from
  FLOOD/DRAIN toward LOCK (addressee-constraining speech as the linguistic form of a costly
  concession);
- attenuated agent (ω_i < 1): for ASSERT, move mass g·(1 − ω_i) from LOCK toward FLOOD and
  from EDGE toward VOID (repetition and withdrawal);
- SHIFT replaces the emitted label with probability g·0.5 when the agent's public position
  changes (factual act opposite to its previous public position).

At g = 0 labels carry exactly the information of the act code up to emission noise: mechanism
recovery from noiseless labels must then equal recovery from acts (built-in check).

**Annotation noise** — three levels:

1. none (z observed as emitted);
2. engine: z passed through the empirical row-normalised confusion matrix of the typed engine
   on the 169 Tier A cases (label-majority over 3 runs; 129/169 correct; file
   `data/engine_confusion_rownorm_tierA.csv`, recomputed 30.09.2026 from
   `out/eval/engine_md3_combined.json`);
3. baseline: same with the direct-prompt confusion matrix (145/169; `data/baseline_confusion_rownorm_tierA.csv`).

## 8. Observers (mechanism recovery)

An idealised observer knows the model and all auxiliary parameters and, for each agent,
compares two families on the same grids used by the agents: attenuation {ω ∈ grid, c = 0} and
concession cost {ω = 1, c ∈ grid}; the reference point (1, 0) belongs to both. Observation
channels: (a) both agents' acts + both private records; (b) both agents' acts only;
(c) speech labels only, under each noise level. For channels (b) and (c) the observer filters
exactly over the unobserved private records. Reported: correct-family rate, mean posterior on
the correct family, parameter MAE against the prior-mean baseline.

## 9. What is observed, inferred, assumed

| quantity | status |
|---|---|
| acts u_t, public positions, labels z_t | observed by agents / observers |
| x, partner's s, c, ω | inferred by agents (joint posterior) |
| own ω_i, c_i, k, b, τ, β, λ, ρ, N | assumed known to the agent; auxiliary for observers |
| emission E0, modulation g, noise matrices | stipulated (E0, g) or empirical (noise) |
| level-0 model of the partner | assumed by each agent; false by construction |

Nothing in this model is a claim about people; the label layer tests identifiability under an
explicit observation model, not the validity of the annotation scheme.
