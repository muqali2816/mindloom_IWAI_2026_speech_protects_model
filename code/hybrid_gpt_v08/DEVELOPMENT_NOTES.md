# Development and interpretation record

- The user requested a hybrid of M-bridge and reciprocal v0.7. The protocol was
  specified for this implementation and is exploratory, not preregistered.
- Planning uses a bounded immediate-verification/next-partner-response window.
  It avoids calling a posterior shrink an information gain. Actual agents are
  level-1; their partner model is cost-only level-0 and deliberately imperfect.
- Changed action names to STATE_0/STATE_1 so switching costs follow current
  public commitment after a concession. PRESSURE is represented by restriction
  form rather than a redundant nonfactual action.
- Main observer uses four nonduplicated focal profiles per participant; the
  eight-point cube is used in participants' beliefs. After implementation the
  external observer was generalized to accept an alternative grid; the default
  four-point numerical calculation is unchanged. No eight-point external-grid
  recovery result is claimed.
- Added post-run paired contrasts for gamma versus reference with and without
  information value, and cost versus reference. This reruns five selected
  dynamics conditions on exactly the original worlds. It changes no parameters.
- Raw LOCK and SEEK output supports are disjoint, which makes that historical
  two-row channel unusually discriminative. Jeffreys-style +0.5 per cell is an
  explicit prior choice; Dirichlet uncertainty is not a domain-transfer model.
- After inspecting dynamics, the report explicitly separates internal belief
  revision, public disagreement and access to evidence. The high-cost pair
  retains public disagreement but does not lose factual learning in this setup.
- The observer's sample of 96 worlds (24 per A profile, B reference) is a pilot.
  Mechanism recovery, next-act prediction and arrival prediction are reported
  separately, including null or adverse results. No tuning to recovery scores.

- Added a descriptive prior sensitivity check for Dirichlet per-cell counts
  0.1, 0.5 and 1.0 (channel information only, not a new observer evaluation).
  This changes no main settings and is reported as an assumption sensitivity.

- After all 96 main observer worlds completed, the sensitivity stage was
  resumed with four independent local processes to reduce wall time. Seeds,
  world indices, path counts and inference arithmetic are unchanged. The
  interruption in observer.log is a scheduling checkpoint, not a model error.

- Corrected metadata for the observed-form observer to paths=1; hidden-form
  runs remain K=8. This changes no posterior, prediction or result metric.
- Final report: eleven pages visually inspected; tables and text fit.
