# Hybrid reciprocal verification dyad v0.8

Exploratory model and computational validation, 30 September 2026. This is a
new specification, not a numerical continuation of either earlier table.
No human parameters, text generation, clinical classification or physiology.

The updating studied here is Bayesian revision of a factual belief and inference
over a fixed dictionary of partner profiles. The likelihood structure, action
vocabulary and dictionary themselves are not learned. Thus this is not yet a
model of long-term acquisition or structural revision of a generative model.

## What is combined

From M-bridge: joint choice of a coarse act and an interactional form;
invitation/restriction controls subsequent access to verification; a separate
cost gamma for publicly disconfirming evidence; normalized factual likelihoods;
labels measure forms, not mechanism parameters.

From reciprocal v0.7: asymmetric agents, initial private records, partner speech
as evidence, level-1 inference of partner mechanisms, and temporary reassurance.
The manual marginal shrink, static SHIFT decoder, global placebo, and inactive
public-position state of v0.7 are not part of this hybrid.

## 1. World and timing

A static binary fact x has physical prior 1/2. Each agent receives two independent
private records with reliability rho_private=.75, conditional on x. Their counts
s_A,s_B are correlated through x. A initially favours 1 and B favours 0 with prior
probability .75; both incorporate their own initial records at rho_private.

Eight rounds contain sixteen alternating moves (A,B,A,B,...). At each event:

1. Speaker i selects y=(u,f) from five acts times two forms.
2. Listener j observes the act and its form and updates its joint posterior.
3. The speaker updates its public position and the record of messages it sent.
4. With probability lambda(y_i,y_j,last), one fresh verification outcome arrives.
5. Both participants observe this same outcome exactly once, then roles alternate.

Initial partner moves are SILENCE with invitation. The private data differ between
participants. Later verification is common; it is not duplicated into independent
private observations. The physical process never reveals x directly to participants.

## 2. Acts, forms and public commitment

u in {STATE_0, STATE_1, ASK, SILENCE, REASSURE}; f in {0 restrict,1 invite}.
The factual act explicitly names a version. A concession is a factual act that
changes the current public position; it is not a permanently named action code.
This makes costs and public memory causally consistent after a switch.

PRESSURE is not retained as a redundant copy of SILENCE. Restriction is represented
by the form f. These are idealized interactional actions, not generated sentences.

lambda_base=.9 if either current/last move is ASK, otherwise .35.
lambda = .05 + (lambda_base-.05) f_i f_j,last.
The no-gating control sets lambda=lambda_base. The form-to-access relationship is
stipulated and requires empirical investigation.

## 3. Three independent parameters

theta_i=(omega_i,c_i,gamma_i), separately set for each participant.

- omega controls subjective reliability of NEW verification outcomes:
  rho_omega = sigmoid(omega logit(rho_verification)). Arrival probabilities are
  never tempered. Initial private evidence uses rho_private, unchanged by omega.
  Partner speech is interpreted with a generative model, not a manually shrunk
  posterior. Therefore omega is not a generic sensitivity to every information source.
- c is the cost of switching the current public position. REASSURE from the last
  partner move sets c_eff=0 for the recipient's next choice only. It does not erase
  earlier beliefs and does not remove gamma.
- gamma prices an expected verification outcome contradicting the public position
  that would hold AFTER the proposed act. It is not anchored forever to the initial
  position, and it is not biological maintenance expenditure.

Four focal profiles: reference=(1,0,0); attenuation=(.3,0,0);
concession cost=(1,1.4,0); access protection=(1,0,2).
Partner inference considers all eight combinations omega in {.3,1}, c in {0,1.4},
gamma in {0,2}. The physical sensor reliability is .75.

## 4. Bayesian inference about the partner

Each actual agent is level-1 and models its partner as level-0: the modelled partner
selects by expected costs only. This mismatch is intentional and limits interpretation.
There is no recursive equilibrium or recovery of an unbounded theory of mind.

J_i(x,s_j,theta_j) starts as P_i(x|s_i) P(s_j|x) Uniform(theta_j).
The modelled partner's belief has log-odds:

  prior_j + (2s_j-N) logit(rho_private)
  + signed_common_verifications logit(rho_omega_j)
  + signed_messages_sent_by_i omega_j logit(trusted_message_precision).

trusted_message_precision=.7. The level-0 approximation treats received assertions
as fixed-precision messages. The ACTUAL level-1 agent infers a static private count
from the partner's policy; it does not add each assertion as a new private record.
Known common verification outcomes and earlier sent messages are conditioned on.

After partner move y_j:
  J_i(new) proportional to J_i(old) pi_level0(y_j|s_j,theta_j,public history).
After a delivered verification e:
  J_i(new) proportional to J_i(old) A_omega_i(e|x,observed chosen moves).
No-signal likelihood cancels for x after the chosen moves are known. There is no
manual posterior adjustment. Inferring parameters can correlate with inferring x;
no claim is made that parameter marginals are unaffected by belief updates.

With finite private records the cumulative log evidence about x from partner speech,
after subtracting known shared-verification updates, is bounded by
N*logit(rho_private), regardless of repeated assertions. This is checked numerically.

## 5. Action selection and its exact information term

For a candidate y_i, let v_i(y_i) be the position after it. For a factual act:
ordinary loss = k P_i(x != v_i) + c_eff 1[v_i != previous position].
For other acts ordinary loss=b. k=2, b=1.1.

Expected cost R_i(y_i) adds restriction_cost*(1-f_i), restriction_cost=.12, and
gamma_i * lambda * P_i(next verification contradicts v_i | a signal arrives).
The contradiction probability is computed using the same subjective rho_omega_i.

The bounded planning window is immediate E, followed by the partner's next Y_j,
ending BEFORE the verification after that response. The level-0 partner updates
on the proposed factual act and on E, then responds with its cost-only policy.

For latent l=(x,s_j,theta_j):
  p(E,Y_j|l,y_i)=A_omega_i(E|x,y_i,last_y_j)
                 pi_level0(Y_j|s_j,theta_j,history,y_i,E).
These probabilities normalize across 3*10 possible outcome pairs for each l,y_i.

IG_i(y_i)=I_{J_i}(l ; (E,Y_j) | y_i).
G_i(y_i)=R_i(y_i)-beta*IG_i(y_i).
pi_i(y_i)=softmax(-G_i(y_i)/tau), tau=.2, main beta=1.

The implementation computes H(E,Y_j)-E_l H(E,Y_j|l). An independent scalar
implementation computes the expected log-likelihood ratio and matches it.
This is the extrinsic-minus-epistemic form of a discrete active-inference objective;
it is also mathematically equivalent to expected utility plus information value.
No unique superiority over that equivalent formulation is claimed.
The implementation fixes the policy temperature and uses explicit expected losses;
it does not infer policy precision or implement a neuronal process theory.

Controls: beta=0 (cost-only actual agents) and verification-only IG (no epistemic
value from the next partner response). The joint IG targets knowledge of the
partner's private data and parameters as well as x; maximizing it need not maximize
truth accuracy alone.

## 6. Observation channel

The hidden true feature is the interactional form f, with candidate ontology
anchors restriction->LOCK and invitation->SEEK. No eight-regime latent process,
SEAL function or SHIFT event is introduced. The eleven historical output names
are possible noisy classifier outputs, including mistakes; an output named SHIFT
does not imply an actual generated transition event.

For each form row, alpha_z = historical count_z + .5. The rows use 15 LOCK and
20 SEEK cases from the archived primary-label confusion matrix. The original
169 cases include several output types; they are not all feature-level Tier A.

C_f ~ Dirichlet(alpha_f). One C is drawn per posterior-predictive world and is
fixed throughout that world's dialogue, shared by both participants. This averages
over uncertainty about calibration; it does not assert that the engine physically
changes after each dialogue. There is no added fixed 25% uninformative mixture.

z_t ~ C_{f_t}. C is independent of theta given f. The observer integrates each
row using its Dirichlet sufficient counts conditional on a proposed latent-form
history. Counts include both speakers' labels. This models calibration sampling
uncertainty, not domain shift, ontology validity or real dialogue annotation skill.

## 7. External observer and honest approximation boundaries

The main external dictionary has four mutually exclusive focal parameter points,
with a uniform prior for each participant. Both A and B parameters are unknown.
The generated observer sample balances the four A profiles and holds B at the
reference profile. It is not an evaluation on all sixteen generating dyad types.
There is no duplicated reference point. The code accepts an arbitrary alternative
grid, including the eight-point cube, but main results are for the four-point grid.

Each of the 16 parameter pairs is crossed with 9 private-count pairs. The correct
physical prior sums over their shared x. Both speakers' true level-1 choice
likelihoods enter the update. Shared verification uses the physical sensor model
for the external observer, distinct from the agents' possibly attenuated models.

Channels all include coarse acts and already delivered verification outcomes:
- forms_evidence: additionally exact forms; finite-grid posterior is exact.
- actions_evidence: forms hidden.
- labels_evidence: additionally noisy form labels, available before verification.

For hidden forms, K path particles are maintained WITHIN EACH static
parameter/private stratum. Strata are never dropped. A fully adapted proposal
conditions on the current coarse act, current label if available, and verification
outcome; form histories are then resampled within strata as necessary. Dirichlet
rows are Rao-Blackwellized with path-specific counts. This is approximate, not an
exact filter; K sensitivity and short exact path enumeration are reported.

Forecasts: the next coarse act is scored using only the previous prefix. The
immediate verification-arrival forecast uses the just-observed coarse act and its
annotation, before the outcome arrives. No latent truth or private beliefs are
exposed. Private counts are marginalized, not observed.

## 8. Intervention semantics

The forced REASSURE/sham/SILENCE event is index 7, i.e. B's move in round 4.
All are invitations. Sham disables only that move's cost relief; both participants
know no relief was granted. This is a controlled inactive message, not a deceptive
psychological placebo. No earlier spontaneous reassurance is changed.

Permanent removal of c or gamma from that event changes participants' actual
parameters. Their partner models still use the original finite static dictionary;
this is a controlled parameter change, not an inferred human treatment effect.
Forcing invitation changes form without changing the chosen coarse act. The
optional bypass overrides actual access externally; it is not used in main results.

## 9. Roadmap

Measure beliefs independently of speech, randomize availability/reliability and
public cost, annotate only the available prefix, and test held-out next exchanges.
Estimate the effect of invitation/restriction rather than assuming it. Calibrate
the two-form measurement on the target corpus, then ask whether richer ontology
features add prediction. Test broader mechanism dictionaries and level-1 model
misspecification. Physiological/allostatic load requires a separate measurement
model and data; no dimensionless cost here is an allostatic-load measurement.
