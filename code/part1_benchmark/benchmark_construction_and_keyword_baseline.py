"""Benchmark construction statistics and a rule-based (keyword) baseline for the 169 Tier A cases.

The case texts are NOT part of this repository (author's decision); this script reads them from a local
directory of Markdown case files (front matter + '## Document text'), e.g. the engine repository's
data/gold/heldout_v2_external_en/. Gold labels are taken from the archived engine JSON so that the
baseline is scored on exactly the same cases and labels as the engine and the direct prompt.

Outputs (results/):
  benchmark_construction.csv  – case types, lengths, sources, exact and near duplicates (word 5-gram Jaccard)
  part1_keyword_baseline.csv  – strict accuracy, macro-F1, SHIFT recall of (i) hand-written rules derived from
                                 the label definitions in Table 1 and REVISED THREE TIMES after inspecting their
                                 output on the same 169 scored cases (see RULES revision history) – the rule score
                                 is therefore an optimistic estimate, not an independent held-out baseline;
                                 (ii) majority-class and (iii) uniform-random baselines
  part1_confusion_keyword.csv – confusion counts of the rule baseline

Usage: python code/part1_benchmark/benchmark_construction_and_keyword_baseline.py --cases_dir <dir>
"""
import re, json, argparse, itertools, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
LABELS = ['BUILD', 'SEEK', 'UNSEAL', 'LOCK', 'DRAIN', 'FLOOD', 'EDGE', 'VOID', 'SEAL', 'SHIFT', 'INSUFF']; GOLD = LABELS[:10]


def load_cases(cases_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(cases_dir) / '*.md'))):
        s = Path(f).read_text(); fm = yaml.safe_load(s.split('---')[1])
        m = re.search(r"## Document text\s*\n(.*?)(?:\n## |\Z)", s, re.S)
        txt = " ".join(l.lstrip("> ").strip() for l in m.group(1).splitlines() if l.strip()) if m else ""
        rows.append(dict(id=fm.get('case_id'), tier=fm.get('tier'), source=fm.get('source'), doc_type=fm.get('document_type'), text=txt, n_words=len(txt.split())))
    return pd.DataFrame(rows)


def shingles(t, n=5):
    w = re.findall(r"\w+", t.lower()); return set(zip(*[w[i:] for i in range(n)])) if len(w) >= n else {tuple(w)}


def construction_stats(df):
    S = [shingles(t) for t in df.text]; rows = []
    for tier in sorted(df.tier.unique()):
        idx = df.index[df.tier == tier].tolist(); sub = df.loc[idx]
        pairs3 = sum(1 for i, j in itertools.combinations(idx, 2) if S[i] and S[j] and len(S[i] & S[j]) / len(S[i] | S[j]) >= 0.3)
        pairs2 = sum(1 for i, j in itertools.combinations(idx, 2) if S[i] and S[j] and len(S[i] & S[j]) / len(S[i] | S[j]) >= 0.2)
        rows.append(dict(tier=tier, n_cases=len(sub), doc_types=";".join(sorted(sub.doc_type.unique())), sources=";".join(sorted(sub.source.unique())),
                         words_min=int(sub.n_words.min()), words_median=float(sub.n_words.median()), words_max=int(sub.n_words.max()),
                         exact_duplicate_texts=int(sub.text.duplicated().sum()), near_dup_pairs_J5_ge_0_3=pairs3, near_dup_pairs_J5_ge_0_2=pairs2))
    return pd.DataFrame(rows)


# ---- rule baseline: patterns written from the Table 1 definitions; precedence order fixed a priori. The rules were
# then revised three times after observing match counts / a confusion matrix on the 169 scored cases (no held-out set
# exists), so the reported accuracy is tuned on the test cases and is an upper bound for a rule system of this kind.
# Revision history (disclosed): v1 FLOOD rule (any repeated word or two absolutist words) fired on 131/169 cases and was
# tightened to >=2 exclamation marks, an all-caps word (case-sensitive sub-pattern), or >=3 absolutist words; v2 SHIFT rule
# ('wait' anywhere) matched 'what can wait' and was restricted to clause-initial turn markers; v3 SEEK rule (bare wh-words anywhere) absorbed 144/169 cases and was restricted
# to question marks and explicit request/inquiry phrases. Rules were frozen after v3; the reported numbers are v3.
RULES = [
    ('SHIFT', r"(^|[.!?]\s+|—\s*|\.{3}\s*)(wait|hold on|actually,|no,? actually|you'?re right|i was wrong|okay,? fine,? you|then again|on second thought)\b"),
    ('LOCK', r"(not allowed|don'?t get to|you can'?t (say|question|bring)|don'?t (you )?dare|not up for (debate|discussion)|no (more )?questions|you will (not|never)|stop (questioning|asking|bringing))"),
    ('SEAL', r"(topic is closed|we'?re done|end of (the )?(discussion|conversation)|i'?m done (talking|discussing)|not (going to|gonna) (talk|discuss)|no point (in )?(continuing|discussing)|this conversation is over|closed\b)"),
    ('DRAIN', r"(i need you to|you have to (help|fix|save|be here)|only you can|can'?t do this without you|need you (here|to be)|it'?s on you to|you'?re the only one)"),
    ('UNSEAL', r"\bi(?:'m| am| feel| felt| was)\b.*\b(scared|afraid|hurt|ashamed|lonely|alone|anxious|terrified|vulnerable|sad|overwhelmed|exhausted|nervous)\b"),
    ('VOID', r"^(whatever|fine|forget it|it doesn'?t matter|nothing|doesn'?t matter|never mind|i don'?t care)\b|\b(it doesn'?t matter|forget it|whatever you say|nothing to say)\b"),
    ('EDGE', r"(come (back|here).*\b(go|leave)\b|\b(leave|go)\b.*\b(don'?t (go|leave)|stay|come back)\b|i want you (close|here).*\bbut\b|\bbut\b.*\b(not too close|back off|need space)\b|(i miss you|come closer).*(get away|need space|back off))"),
    ('FLOOD', r"(!{2,}|(?-i:\b[A-Z]{4,}\b)|(?:\b(?:always|never|everything|nothing|every single|all the time|constantly|every time)\b.*){3,})"),
    ('SEEK', r"\?|\b(could you|can you|would you|tell me|help me understand|i (asked|want to know|wonder)|i'?m (asking|curious)|do you (think|know))\b"),
    ('BUILD', r"\b(let'?s|we (can|could|might)|what if we|together|i'?ll|i (split|made|built|set up|sorted)|plan|next step|step by step|one at a time)\b"),
]


def rule_predict(text, fallback='SEEK'):
    t = text.strip()
    for label, pat in RULES:
        if re.search(pat, t, flags=re.I): return label
    return fallback


def macro_f1(M):
    f1 = []
    for g in GOLD:
        tp = M.loc[g, g]; fp = M[g].sum() - tp; fn = M.loc[g].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.; r = tp / (tp + fn) if tp + fn else 0.
        f1.append(2 * p * r / (p + r) if p + r else 0.)
    return float(np.mean(f1))


def score(preds, gold, name):
    M = pd.DataFrame(0, index=LABELS, columns=LABELS)
    for g, p in zip(gold, preds): M.loc[g, p] += 1
    ns = [(g, p) for g, p in zip(gold, preds) if g != 'SHIFT']
    return M, {'system': name, 'n_cases': len(gold), 'strict_acc': float(np.mean([g == p for g, p in zip(gold, preds)])),
               'non_shift_acc': float(np.mean([g == p for g, p in ns])), 'macro_f1': macro_f1(M),
               'shift_recall': float(M.loc['SHIFT', 'SHIFT'] / max(M.loc['SHIFT'].sum(), 1))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cases_dir', required=True); args = ap.parse_args()
    df = load_cases(args.cases_dir)
    construction_stats(df).to_csv(OUT / 'benchmark_construction.csv', index=False)
    eng = json.loads((ROOT / 'data' / 'archived_predictions' / 'engine_md3_combined_2026-07.json').read_text())
    gold = {}
    for c in eng['per_run_outcomes'][0]:
        rows = c.get('primary_segment_results') or []
        if rows: gold[c['case_id']] = rows[0]['expected'].split('.')[-1]
    A = df[df.id.isin(gold)].copy(); A['gold'] = A.id.map(gold)
    assert len(A) == 169, len(A)
    out = []
    M, r = score([rule_predict(t) for t in A.text], A.gold.tolist(), 'keyword_rules'); out.append(r); M.to_csv(OUT / 'part1_confusion_keyword.csv')
    maj = A.gold.value_counts().idxmax(); out.append(score([maj] * len(A), A.gold.tolist(), f'majority_class({maj})')[1])
    rng = np.random.default_rng(0); out.append(score(list(rng.choice(GOLD, len(A))), A.gold.tolist(), 'uniform_random(seed0)')[1])
    pd.DataFrame(out).to_csv(OUT / 'part1_keyword_baseline.csv', index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False)); print(construction_stats(df).to_string(index=False))
