# Hybrid reciprocal verification dyad v0.8

Exploratory research prototype, 30 September 2026, for the Mindloom manuscript
**Speech that Protects a Model**. Combines the M-bridge act/form/access mechanism
with reciprocal private-evidence inference and reassurance from the v0.7 idea.
The implementation is new. Original manuscript and archived v0.7 are unchanged.

Read `MODEL_SPEC.md` for the exact process, timing, information objective,
observation model and limitations. The accompanying Russian report explains
what the results support and what remains untested. `report/Proposed_hybrid_sections_EN.txt`
contains an English manuscript insertion, not an edited full manuscript.

## Reproduce

Python 3.11+; NumPy, SciPy and pandas. Report additionally uses matplotlib and
reportlab; PDF inspection uses PyMuPDF. Exact tested versions are recorded in
`environment.json`. No API calls, language model, GPU or external credentials.

```bash
python3 validate_independent.py
python3 run_dynamics.py
python3 run_observer.py
python3 analyze_results.py --additional-dynamics
python3 analyze_results.py --observer
python3 build_report.py
```

The observer is the expensive stage: it runs 96 separate dialogues with hidden
form histories and a path-count sensitivity analysis. On this workspace it takes
tens of minutes, not seconds. Do not interpret a partial `observer_worlds.csv` as
final. Completed stages have `*_manifest.json` files. All random seeds and
baseline settings are in `protocol.json`; changing it changes the experiment.
Reproduction runs overwrite results in this project only.

## Project map

- `model.py`: two actual level-1 agents, ten act×form moves, shared verification,
  private initial records, current public commitment, three mechanism parameters.
- `observer.py`: exact static-parameter/private-count strata, exact observed-form
  filter, approximate hidden-form filter, optional short exhaustive path filter.
- `channel.py`: historical two-row counts plus Dirichlet sampling uncertainty.
- `validate_independent.py`: independent scalar arithmetic and causal controls.
- `run_dynamics.py`: 2,000 common worlds; sixteen mixed parameter pairs; ablations
  and targeted interventions; four sensor-reliability sensitivity conditions.
- `run_observer.py`: 96 independent worlds; labels are available only after an
  act and before verification; no future information enters predictions.
- `run_sensitivity.py`: independently seeded local worker processes for the
  K = 4/32 observer checks; can resume after a completed main observer stage.
  A full run_observer.py run resets these checks. Standalone resume assumes
  unchanged code, protocol and main observer outputs.
- `analyze_results.py`: paired reporting contrasts and numerical sensitivity.
- `results/`: tables, calibration posterior, held-out records, validation logs.
- `source/`: unchanged historical confusion count tables copied from the
  supplied reciprocal v0.7 archive. Baseline counts retained for provenance;
  only engine LOCK/SEEK rows enter the new channel.
- `report/`: report, English insertion and figures.

`protocol.json` is an exploratory development record, not a registered protocol.
`DEVELOPMENT_NOTES.md` records later checks and interpretation changes. The archive
hash manifest covers the delivered files; it is not independent replication.

## Three interpretive boundaries

1. A persistent public position does not imply unchanged private belief.
2. Simulated speech labels measure a stipulated interactional form, not a
   person's motive, allostatic burden or independently validated mechanism.
3. The information term is exact mutual information under a bounded partner
   model. A cost-plus-information formulation is equivalent; this work does not
   establish a unique advantage for active inference as a theoretical label.

The task differs from both previous models (timing, initial data, actions,
parameters, observer dictionary). Its numerical scores cannot replace an old
Table 5 row without changing the table and methods description.
