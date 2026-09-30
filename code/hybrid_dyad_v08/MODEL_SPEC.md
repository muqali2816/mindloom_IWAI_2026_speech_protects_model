# Hybrid dyad v0.8 — speech as evidence *and* as control of joint verification

Status: specification written before implementation and before any run (30.09.2026). Merges
M‑bridge (Mindloom_dyadic_bridge_2026‑09‑30: act × form, joint forms gate shared evidence, γ) with
v0.7 (reciprocal_dyad: level‑1 inference about the partner from the partner's acts, REASSURE,
per‑agent mechanisms). Both parents are regression anchors (§8). Simulation of stipulated
mechanisms; no human data.

## 1. World

Binary fact x, balanced. Agents A (prefers version 1) and B (version 0). Prior ±logit(0.75).
Each agent holds N private records (default N = 2; N = 4 reproduces v0.7, N = 0 reproduces M‑bridge)
of reliability ρ = 0.75. R = 12 rounds. Default move order: **simultaneous** (both agents choose
from the state at the start of the round, then both observe both moves); 'AB'/'BA' sequential
orders are available.

## 2. Per‑agent mechanisms

| symbol | meaning | parent |
|---|---|---|
| ω_i ∈ [0,1] | evidence sensitivity to everything the dialogue delivers: log‑odds increments from shared signals and from the partner's acts are scaled by ω_i (marginal shrink; for a shared signal this equals the arrival‑preserving ρ_ω of M‑bridge) | both |
| c_i ≥ 0 | additional loss of each public CONCEDE; relieved for one move by the partner's REASSURE | both |
| γ_i ≥ 0 | aversion to expected public contradiction: γ_i · λ̂ · d_i, d_i = subjective probability that the next shared signal contradicts i's **initial** position | M‑bridge |
| β | weight of expected information gain | both |

Types (protocol.json): honest (1,0,0), attenuated (0.2,0,0), costly (1,1.6,0), access‑protecting (1,0,2), mixed (0.6,0.8,1).

## 3. Options: act × form

Acts: ASSERT, CONCEDE, PRESSURE, SILENCE, ASK, REASSURE. Forms: f = 1 invite verification, f = 0 restrict it.
Option index k = 2·u + f (u‑major), drawn by inverse CDF on one uniform — the M‑bridge convention.

Loss ℓ_i(u): k(1 − q_own), k·q_own + c_i·(1 − relief_i), b, b, b, b (k = 2, b = 1.10). Restriction cost δ = 0.12.

## 4. Shared verification channel (M‑bridge)

At the start of round t a shared signal arrives with probability
λ_t = λ_floor + (λ_base(u_A,t−1, u_B,t−1) − λ_floor) · f_A,t−1 · f_B,t−1,
λ_base = 0.90 if either previous act was ASK, else 0.35; λ_floor = 0.05. Signal e ∈ {0,1} with
P(e = x) = ρ. Both agents update: logit q_i += ω_i · (2e − 1) · llr(ρ). Round 0 uses previous
forms = invite, previous acts = ASSERT.

## 5. Level‑1 inference about the partner (v0.7)

Agent i holds J_i(x, s_j, c_j, ω_j) on grids c ∈ {0, 0.8, 1.6}, ω ∈ {0.2, 0.6, 1.0}. The partner is
modelled as **level‑0 over acts only** (no forms, no γ): logit q_j = prior_j + EV(s_j) + ω_j·(M_i + m_t·llr(ρ)),
where M_i is the log‑odds shift i's own assertions would induce at trusted precision λ = 0.70 and
m_t the signed count of shared signals; P(u_j | cell) = softmax(−ℓ_j/τ) over the six acts with
c_j relieved if i reassured. After observing u_j, J_i is updated exactly and its x‑marginal is
shrunk by ω_i (v0.7 rule). The level‑0 partner model deliberately omits forms so that the agent's
internal state depends only on public acts, shared signals and relief — the condition under which
the external observer of §7 is exact.

## 6. Policy

G_i(u,f) = ℓ_i(u) + δ(1 − f) + γ_i · λ̂_i(u,f) · d_i − β · [ IG_i^{partner}(u) + λ̂_i(u,f) · I_bin(q_i, ρ_{ω_i}) ],

λ̂_i(u,f) = λ_floor + (λ_base(u, u_j,prev) − λ_floor) · f · f_j,prev (partner's previous form held
fixed — M‑bridge forecast); IG^{partner} = expected entropy reduction of J_i from the partner's
next act (v0.7); I_bin = mutual information of the binary signal under subjective reliability
ρ_ω = ρ^ω/(ρ^ω + (1−ρ)^ω). P(u,f) ∝ exp(−G/τ), τ = 0.20.

## 7. Observation model and observers

**Labels.** z_i,t depends on the form only: rows LOCK (restrict) and SEEK (invite) of the archived
169‑case confusion matrix (15 and 20 cases). Generation samples each world's two rows from
Dirichlet(counts + 0.5) — row uncertainty is integrated rather than fixed by a 25 % mixture; the
observer uses the posterior‑mean rows, optionally mixed with a state‑independent background at
level η ∈ {0, 0.25, 0.5} (transport stress test, as in M‑bridge).

**Observers** (exact, known model): hidden state = (x, s_A, s_B, previous form pair); the agents'
internal posteriors are replayed deterministically for each (hypothesis, s); the HMM uses the
arrival likelihood of shared signals, both agents' joint act–form policies and, per channel,
(a) acts + signal record, (b) + form labels, (c) + oracle forms, (d) + null labels that depend
only on the observed act (must give zero gain). Hypotheses for agent A: attenuation {ω ∈ grid, c = γ = 0},
concession cost {ω = 1, c ∈ grid, γ = 0}, access protection {ω = 1, c = 0, γ ∈ {1, 2, 4}}, reference (1,0,0)
shared. Outputs: family posterior, parameter posterior means vs prior means, next‑signal and
next‑act log loss.

## 8. Regression anchors (must hold before any experiment)

| configuration | reproduces |
|---|---|
| level‑1 off, N = 0, acts {ASSERT, CONCEDE, PRESSURE, SILENCE, ASK}, forms on, order 'sim', M‑bridge worlds | M‑bridge `simulate` for the 4 hypotheses and interventions: acts, forms, signals identical; beliefs ≤ 1e‑10 |
| level‑1 on, N = 4, acts {ASSERT, CONCEDE, PRESSURE, SILENCE, REASSURE}, forms off, λ ≡ 0, γ = 0, order 'AB', v0.7 worlds | v0.7 `dialogue`: acts identical, beliefs ≤ 1e‑10 |

Further checks: ω = 0 → no belief change; c → ∞ → no unrelieved concession; role‑swap symmetry
under 'sim'; brute‑force enumeration of short form paths = HMM filter; null‑label and
coupling‑off channels give zero gain; seed repeatability; scalar re‑implementation of one policy.

## 9. Observed, inferred, assumed

| quantity | status |
|---|---|
| acts, forms (to the partner), shared signals, labels (to the observer) | observed |
| x; partner's s, c, ω | inferred by each agent |
| own ω, c, γ; k, b, δ, τ, β, λ's, ρ, N | assumed known to agents; auxiliary for observers |
| effect of forms on λ; rows LOCK/SEEK as the label channel of forms | stipulated (process) / empirical rows with assumed transport |
| level‑0, form‑free model of the partner | assumed by each agent; false by construction |
