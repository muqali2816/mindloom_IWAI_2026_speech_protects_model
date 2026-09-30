# v0.7 (2026-09-30) — reciprocal level-1 dyad + speech-label observation layer
- Both agents level-1 with (omega, c); joint posterior over partner's (s, c, omega); five acts incl. REASSURE; one-step EFE with IG.
- Attenuation rule: marginal-shrink on the interlocutor channel (changed from likelihood tempering before any experiment; MODEL_SPEC §4, protocol.amendments).
- Speech-label layer: E0 emission, modulation g, SHIFT on position change, empirical confusion noise (engine 129/169, direct prompt 145/169).
- observer.py: family recovery from acts+private / acts / labels (plug-in decoder).
- validate.py: 10 checks incl. bitwise regression to v0.6 (acts identical, beliefs <= 4e-14).
- Experiments E1–E5 (2000 worlds each, seeds 20261001–20261005) by three tracks; scripts/, results/, figures/, methods/.
