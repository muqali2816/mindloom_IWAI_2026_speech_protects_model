#!/usr/bin/env python3
"""Ablation scorer: naive LLM baseline vs full engine, identical metrics.

Reads eval JSONs in the standard `per_run_outcomes` schema (the same artifact
the engine harness writes and `primary_confusion.py` consumes). Computes, on
the SAME cases / gold labels / scoring rules:

  - strict accuracy (all classes, and non-SHIFT)
  - macro-F1 (all classes, and non-SHIFT)
  - per-class precision / recall / F1
  - primary confusion matrix (top off-diagonal)
  - manipulation-gate accounting -> fabricated-bridge rate (L3)

The point of the ablation is that the delta (full - naive) on these exact axes
attributes value to the architecture (typing + gates), not to a different
method. Feed the engine artifact and the baseline artifact; the tool prints
each one's metrics and the paper comparison table.

Usage:
  # single artifact
  python scripts/baseline_compare.py --engine out/eval/wave5_ext_primary/eval_*.json --lang en

  # comparison (paper table)
  python scripts/baseline_compare.py \
      --engine   out/eval/wave5_ext_primary/eval_*.json \
      --baseline out/eval/naive_baseline/eval_*.json \
      --lang en
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field


def short(pid: str) -> str:
    return pid.split(".")[-1] if "." in pid else pid


def _load(path_glob: str) -> dict:
    paths = sorted(glob.glob(path_glob))
    if not paths:
        raise SystemExit(f"no file matches {path_glob}")
    return json.load(open(paths[-1]))


@dataclass
class Metrics:
    path: str
    n_cases: int
    strict_ok: int
    strict_tot: int
    ns_ok: int
    ns_tot: int
    per_class: dict  # label -> (support, prec, rec, f1)
    macro_f1: float
    macro_f1_ns: float
    confusion: Counter  # (expected, predicted) -> n
    # manipulation gate accounting
    manip_emitted: int = 0
    manip_with_bridge: int = 0
    manip_dropped_no_bridge: int = 0
    manip_dropped_no_cp: int = 0
    neg_cases: int = 0
    neg_spurious: int = 0

    @property
    def strict_acc(self) -> float:
        return self.strict_ok / self.strict_tot if self.strict_tot else 0.0

    @property
    def strict_acc_ns(self) -> float:
        return self.ns_ok / self.ns_tot if self.ns_tot else 0.0

    @property
    def seek_prec(self) -> float:
        v = self.per_class.get("regime.SEEK")
        return v[1] if v else 0.0

    @property
    def fabricated_dropped(self) -> int:
        """Emissions the gate removed as ungrounded (no-bridge + no-counterpart-
        pressure, the latter includes bridge-source-not-in-L1-evidence)."""
        return self.manip_dropped_no_bridge + self.manip_dropped_no_cp

    @property
    def fabricated_ungated_total(self) -> int:
        return self.manip_emitted + self.fabricated_dropped


def _vote_consensus(runs: list[list[dict]]) -> list[dict]:
    """Collapse R independent runs into one per-case median-of-R consensus:
    majority vote on each segment's predicted label (ties -> run-0 value).
    Manipulation-gate counters are averaged->rounded across runs."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for c in run:
            by_id[c["case_id"]].append(c)
    out = []
    for cid, cs in by_id.items():
        base = dict(cs[0])
        rows0 = cs[0].get("primary_segment_results") or []
        voted_rows = []
        for i, r0 in enumerate(rows0):
            votes = Counter()
            for c in cs:
                rows = c.get("primary_segment_results") or []
                if i < len(rows):
                    votes[rows[i]["predicted"]] += 1
            win = votes.most_common(1)[0][0] if votes else r0["predicted"]
            voted_rows.append({**r0, "predicted": win})
        base["primary_segment_results"] = voted_rows
        for k in ("manipulations_emitted", "manipulations_with_bridge",
                  "manipulations_dropped_no_bridge",
                  "manipulations_dropped_no_counterpart_pressure",
                  "spurious_manipulations"):
            vals = [c.get(k, 0) or 0 for c in cs]
            base[k] = round(sum(vals) / len(vals))
        out.append(base)
    return out


def compute(path_glob: str, lang: str | None, vote: bool = False) -> Metrics:
    data = _load(path_glob)
    runs = data["per_run_outcomes"]
    outs = _vote_consensus(runs) if (vote and len(runs) > 1) else runs[0]
    TP: Counter = Counter()
    FP: Counter = Counter()
    FN: Counter = Counter()
    support: Counter = Counter()
    confusion: Counter = Counter()
    strict_ok = strict_tot = ns_ok = ns_tot = n_cases = 0
    m = dict(emit=0, wb=0, dnb=0, dcp=0, negc=0, negfp=0)
    for o in outs:
        if lang and (o.get("language") or "").lower() != lang.lower():
            continue
        rows = o.get("primary_segment_results") or []
        if rows:
            n_cases += 1
        for r in rows:
            e, p = r["expected"], r["predicted"]
            support[e] += 1
            strict_tot += 1
            confusion[(e, p)] += 1
            if e == p:
                strict_ok += 1
                TP[e] += 1
            else:
                FP[p] += 1
                FN[e] += 1
            if short(e) != "SHIFT":
                ns_tot += 1
                if e == p:
                    ns_ok += 1
        m["emit"] += o.get("manipulations_emitted", 0) or 0
        m["wb"] += o.get("manipulations_with_bridge", 0) or 0
        m["dnb"] += o.get("manipulations_dropped_no_bridge", 0) or 0
        m["dcp"] += o.get("manipulations_dropped_no_counterpart_pressure", 0) or 0
        if o.get("is_negative_case"):
            m["negc"] += 1
            m["negfp"] += o.get("spurious_manipulations", 0) or 0

    per_class: dict = {}
    f1s = []
    f1s_ns = []
    for e in sorted(support):
        tp = TP[e]
        prec = tp / (tp + FP[e]) if (tp + FP[e]) else 0.0
        rec = tp / (tp + FN[e]) if (tp + FN[e]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[e] = (support[e], prec, rec, f1)
        f1s.append(f1)
        if short(e) != "SHIFT":
            f1s_ns.append(f1)
    return Metrics(
        path=sorted(glob.glob(path_glob))[-1],
        n_cases=n_cases,
        strict_ok=strict_ok, strict_tot=strict_tot,
        ns_ok=ns_ok, ns_tot=ns_tot,
        per_class=per_class,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        macro_f1_ns=sum(f1s_ns) / len(f1s_ns) if f1s_ns else 0.0,
        confusion=confusion,
        manip_emitted=m["emit"], manip_with_bridge=m["wb"],
        manip_dropped_no_bridge=m["dnb"], manip_dropped_no_cp=m["dcp"],
        neg_cases=m["negc"], neg_spurious=m["negfp"],
    )


def print_metrics(m: Metrics, title: str) -> None:
    print(f"\n{'='*66}\n{title}\n{m.path}\ncases={m.n_cases}  scored_segments={m.strict_tot}\n{'='*66}")
    print("── Per-class precision / recall / F1 ──")
    print(f"  {'class':10s} {'sup':>3s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    for e in sorted(m.per_class, key=lambda x: -m.per_class[x][0]):
        sup, prec, rec, f1 = m.per_class[e]
        print(f"  {short(e):10s} {sup:3d} {prec:6.3f} {rec:6.3f} {f1:6.3f}")
    print()
    print(f"  strict acc (all)       = {m.strict_ok}/{m.strict_tot} = {m.strict_acc:.4f}")
    print(f"  strict acc (non-SHIFT) = {m.ns_ok}/{m.ns_tot} = {m.strict_acc_ns:.4f}")
    print(f"  macro-F1 (all classes) = {m.macro_f1:.4f}")
    print(f"  macro-F1 (non-SHIFT)   = {m.macro_f1_ns:.4f}")
    print(f"  SEEK precision         = {m.seek_prec:.4f}")
    # confusion
    off = [((e, p), n) for (e, p), n in m.confusion.items() if e != p]
    if off:
        print("\n── Top confusions (expected → predicted, off-diagonal) ──")
        for (e, p), n in sorted(off, key=lambda kv: -kv[1])[:15]:
            print(f"  {n:3d} × {short(e):20s} → {short(p)}")
    # manipulation gate
    if m.manip_emitted or m.fabricated_dropped:
        print("\n── Manipulation gate (L3 fabricated-bridge) ──")
        print(f"  emitted post-gate (survivors) = {m.manip_emitted} "
              f"(with bridge = {m.manip_with_bridge})")
        print(f"  dropped no_bridge             = {m.manip_dropped_no_bridge}")
        print(f"  dropped no_counterpart_press. = {m.manip_dropped_no_cp}")
        print(f"  => ungrounded emissions gate removed = {m.fabricated_dropped}")
        print(f"  => full-engine fabricated-in-output = 0 ; ungated would surface {m.fabricated_dropped}")
    if m.neg_cases:
        print(f"\n  negative cases = {m.neg_cases}  spurious manipulations (FP) = {m.neg_spurious}")


def print_comparison(engine: Metrics, baseline: Metrics) -> None:
    print(f"\n\n{'#'*66}\n# PAPER TABLE — controlled ablation (same model/input/gold/scoring)\n{'#'*66}")
    hdr = f"| {'Configuration':28s} | {'strict acc (non-SHIFT)':>21s} | {'macro-F1':>8s} | {'SEEK prec.':>10s} | {'fabricated-bridge':>17s} |"
    sep = "|" + "-"*30 + "|" + "-"*23 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*19 + "|"
    print(hdr)
    print(sep)
    # naive: no bridge machinery -> fabricated-bridge is N/A
    print(f"| {'naive LLM (L0)':28s} | {baseline.strict_acc_ns:21.3f} | {baseline.macro_f1:8.3f} | {baseline.seek_prec:10.3f} | {'—':>17s} |")
    # engine
    print(f"| {'full engine':28s} | {engine.strict_acc_ns:21.3f} | {engine.macro_f1:8.3f} | {engine.seek_prec:10.3f} | {0:17d} |")
    print(sep)
    # deltas
    d_acc = engine.strict_acc_ns - baseline.strict_acc_ns
    d_f1 = engine.macro_f1 - baseline.macro_f1
    d_seek = engine.seek_prec - baseline.seek_prec
    print(f"| {'Δ (engine − naive)':28s} | {d_acc:+21.3f} | {d_f1:+8.3f} | {d_seek:+10.3f} | {'':>17s} |")
    print(f"\nL3 (fabricated-bridge, from engine gate counters): "
          f"full engine = 0, ungated ≈ {engine.fabricated_dropped} ungrounded emissions.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="engine eval JSON (glob ok)")
    ap.add_argument("--baseline", default=None, help="naive baseline eval JSON (glob ok)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--vote", action="store_true",
                    help="collapse multi-run artifacts to per-case median-of-R vote")
    args = ap.parse_args()

    engine = compute(args.engine, args.lang, vote=args.vote)
    print_metrics(engine, "FULL ENGINE")
    if args.baseline:
        baseline = compute(args.baseline, args.lang, vote=args.vote)
        print_metrics(baseline, "NAIVE LLM BASELINE (L0)")
        print_comparison(engine, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
