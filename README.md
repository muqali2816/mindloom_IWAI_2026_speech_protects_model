# Speech that Protects a Model — code, data and manuscript (IWAI 2026)

Companion repository for the paper **"Speech that Protects a Model: Typed Speech Evidence and Belief-Protecting
Mechanisms in Dialogue"** (7th International Workshop on Active Inference, IWAI 2026, Madrid, 14–16 October 2026;
accepted as poster + spotlight). Author: Maria Svet, Mindloom Cognitive Lab.

The paper has two parts and a bridge:

| Part | What it is | Where in this repo |
|---|---|---|
| I  | Typed annotation of protective speech (ten labels: eight configurations BUILD SEEK UNSEAL LOCK DRAIN FLOOD EDGE VOID, one function SEAL, one event SHIFT) and a 589-case held-out benchmark; archived July 2026 outputs of the typed engine and of a direct-prompt baseline on the 169 Tier A cases | `code/part1_benchmark/`, `data/archived_predictions/`, `results/part1_*.csv`, `results/benchmark_construction.csv` |
| II | Single-agent active-inference model of a speaker who keeps a belief unrevised (attenuation ω vs concession cost c), 4000 simulated worlds | **not included** – see `code/part2_single_agent/README.md` (author archive, numbers quoted unchanged from the manuscript) |
| Bridge | Dyadic models in which speech is (a) evidence for the partner and (b) a control on access to joint verification; an external observer with the Part I labels as a noisy channel | `code/mbridge/` (author's M-bridge), `code/reciprocal_dyad_v07/`, `code/hybrid_dyad_v08/` (model used in the paper) |

`paper/` holds the LaTeX sources of the 12-page LNCS camera-ready (`main.tex`, `references.bib`, `main.pdf`);
`docs/` holds the internal review, the model comparison and the two simulation reports (Russian) plus the
English draft sections; `CHANGES_vs_26pp_RU.md` lists every change relative to the 26-page revised version.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# Part I: re-score archived outputs (no model calls), 1 s
python code/part1_benchmark/rescore_archive.py
# Part I: benchmark construction table + keyword baseline (needs the case texts, see below)
python code/part1_benchmark/benchmark_construction_and_keyword_baseline.py --cases_dir <dir with 589 .md cases>
# Bridge: hybrid dyad v0.8 (validation ~1 min; full grid H1–H4 ~40 min on 8 cores)
cd code/hybrid_dyad_v08 && python validate.py --mbridge ../mbridge --v07 ../reciprocal_dyad_v07 && cat README.md
```

The benchmark **texts** (589 Markdown cases) are not redistributed here; the archived prediction files contain
case identifiers, gold labels and predicted labels only. Requests for the texts: see the paper's Data Availability
statement.

## What is and is not claimed

* All bridge results are **simulations of stipulated mechanisms**; no human dialogue data enter Part II or the bridge.
* Part I accuracy numbers (engine 129/169, direct prompt 145/169) are re-scored here from archived model outputs of
  July 2026; the prompts, model version and temperature are listed in `paper/` Appendix B (run configuration).
* The claim that the annotation gate reduced coordinator-noted inconsistencies from 17/65 to 0 is author-reported and
  is **not** recomputable from this repository.

## Layout

```
code/part1_benchmark/        re-scoring, confusion matrices, benchmark construction, keyword/majority/random baselines
code/mbridge/                author's dyadic bridge (dyad_model.py, observer, hypotheses reference/attenuation/cost/access)
code/reciprocal_dyad_v07/    reciprocal dyad (partner's act as evidence; E1–E5 experiments)
code/hybrid_dyad_v08/        hybrid dyad used in the paper (act x form, joint verification channel, level-1 partner model; H1–H4)
code/part2_single_agent/     pointer to the author archive of the single-agent simulator (Tables 4–6 of the paper)
data/archived_predictions/   engine and direct-prompt outputs, July 2026 (ids + labels; no texts)
results/                     CSV tables quoted in the paper
figures/                     figures used in the paper, poster and spotlight slide
paper/                       LaTeX sources (llncs), bibliography, compiled PDF
docs/                        review, model comparison, simulation reports, draft sections
```

## Licence
Code: MIT (`LICENSE`). Manuscript text, figures and documentation: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).
Cite via `CITATION.cff`.
