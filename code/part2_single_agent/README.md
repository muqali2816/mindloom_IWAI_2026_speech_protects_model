# Part II — single-agent simulator (not included)

The single-agent active-inference model of Part II (speaker with private belief q_t over a binary proposition;
actions ASSERT, CONCEDE, PRESSURE, SILENCE, ASK; mechanisms: evidence attenuation ω and concession cost c;
4000 worlds per condition; seeds 20260927–20260930; τ = 0.20, k = 2, b = 1.10, ρ = 0.75, λ0 = 0.35, λ1 = 0.90;
matched conditions (ω, c) = (0.10, 0) and (1, 1.4)) is maintained by the author in a separate archive
(`Mindloom_Defensive_Dialogue_v06`, study seed 2026092721) and was **not** re-run for this repository.

All Part II numbers in the manuscript (Tables 4–6 and the text of Section 3) are quoted unchanged from the
26-page revised version of the paper. The bridge models in `../mbridge`, `../reciprocal_dyad_v07` and
`../hybrid_dyad_v08` re-implement the same per-agent update rules (their `validate.py` scripts reproduce the v0.6
single-agent results bitwise as a regression test on the shared core), so the mechanism definitions used in
the bridge are the ones of Part II.

To obtain the Part II archive, contact the author (see the paper's Data Availability statement).
