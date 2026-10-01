"""Check that every bridge/Part-I number quoted in paper/main.tex matches its source table.

Each entry: (label, value as printed in main.tex, source value, rounding used in the text). The script also greps
main.tex for the printed string so a silent edit of the manuscript is caught. Output: results/manuscript_number_check.csv
and a non-zero exit code if any mismatch. Part II numbers are quoted from the 26-page manuscript and are not checked here.
Usage: python scripts/check_manuscript_numbers.py
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'code' / 'hybrid_dyad_v08' / 'results'
tex = (ROOT / 'paper' / 'main.tex').read_text()
kn = json.loads((R / 'H1_H3_key_numbers.json').read_text())
rec = pd.read_csv(R / 'H2_recovery.csv'); rec = rec[rec.split == 'all'].set_index(['generator', 'channel'])
pg = pd.read_csv(R / 'H2_prediction_gains.csv'); pg = pg[pg.split == 'all'].set_index(['generator', 'channel'])
sc = pd.read_csv(ROOT / 'results' / 'part1_scores.csv').set_index('system')
kb = pd.read_csv(ROOT / 'results' / 'part1_keyword_baseline.csv').set_index('system')
bc = pd.read_csv(ROOT / 'results' / 'benchmark_construction.csv').set_index('tier')
H = kn['H1_A_own0']; L = kn['listener_base_A_own0']; P4 = kn['P4']; N80 = kn['n80_A_own0']

rows = []
def chk(label, printed, source, nd):
    ok = abs(float(source) - float(printed)) <= 0.5 * 10 ** (-nd) + 1e-9   # half-up tolerant
    rows.append(dict(label=label, printed=printed, source=round(float(source), 6), decimals=nd, match=ok, in_tex=(str(printed) in tex)))

# Table 6 (channels, A on own side)
T6 = {'nonc_A': {'honest': '0.794', 'attenuated': '0.981', 'costly': '0.996', 'access': '0.814'},
      'A_dbelief': {'honest': '0.227', 'attenuated': '0.066', 'costly': '0.238', 'access': '0.208'},
      'A_hidden_shift': {'honest': '0.083', 'attenuated': '0.057', 'costly': '0.222', 'access': '0.082'},
      'evidence_count': {'honest': '2.64', 'attenuated': '2.58', 'costly': '2.73', 'access': '2.09'},
      'restrict_A': {'honest': '0.335', 'attenuated': '0.354', 'costly': '0.336', 'access': '0.526'},
      'A_p_truth': {'honest': '0.850', 'attenuated': '0.727', 'costly': '0.859', 'access': '0.828'}}
for m, d in T6.items():
    for t, v in d.items(): chk(f'T6 {m} {t}', v, H[t][m], len(v.split('.')[1]))
for t, v in {'honest': '0.230', 'attenuated': '0.320', 'costly': '0.452', 'access': '0.234'}.items(): chk(f'T6 listener pC_high {t}', v, L[t]['B_pC_high'], 3)
for t, v in {'honest': '0.423', 'attenuated': '0.588', 'costly': '0.468', 'access': '0.435'}.items(): chk(f'T6 listener pW_low {t}', v, L[t]['B_pW_low'], 3)
# access x access
chk('access x access signals', '1.75', kn['P2']['access_access_ev'], 2); chk('honest x honest signals', '2.60', kn['P2']['honest_honest_ev'], 2)
chk('gap diff', '0.017', kn['P2']['gap_diff'], 3); chk('gap diff lo', '0.008', kn['P2']['gap_diff_ci'][0], 3); chk('gap diff hi', '0.025', kn['P2']['gap_diff_ci'][1], 3)
# interventions
c = P4['reassure_minus_placebo__P_concede_A_late']['costly']; chk('reassure costly', '0.155', c['delta'], 3); chk('reassure lo', '0.136', c['lo'], 3); chk('reassure hi', '0.174', c['hi'], 3)
c = P4['remove_c__P_concede_A_late']['costly']; chk('remove c costly', '0.275', c['delta'], 3); chk('remove c lo', '0.251', c['lo'], 3); chk('remove c hi', '0.298', c['hi'], 3)
c = P4['remove_gamma__evidence_count_late']['access']; chk('remove gamma access', '0.199', c['delta'], 3); chk('remove gamma lo', '0.174', c['lo'], 3); chk('remove gamma hi', '0.225', c['hi'], 3)
e = P4['external_obs__A_p_truth']; chk('ext obs attenuated', '0.058', e['attenuated']['delta'], 3); chk('ext obs costly', '0.092', e['costly']['delta'], 3); chk('ext obs access', '0.120', e['access']['delta'], 3)
ic = pd.read_csv(R / 'H3_interventions_contrasts.csv'); ic = ic[(ic.split == 'A_own0')].set_index(['generator', 'contrast', 'metric'])
chk('remove gamma late restriction', '-0.187', ic.loc[('access', 'remove_gamma-base', 'restrict_A_late'), 'delta'], 3)
chk('force open access signals', '0.127', ic.loc[('access', 'force_open_B-base', 'evidence_count_late'), 'delta'], 3)
others = sorted(ic.loc[(t, 'force_open_B-base', 'evidence_count_late'), 'delta'] for t in ('honest', 'attenuated', 'costly'))
chk('force open min other', '0.31', others[0], 2); chk('force open max other', '0.33', others[-1], 2)
chk('force open access late restriction', '0.067', ic.loc[('access', 'force_open_B-base', 'restrict_A_late'), 'delta'], 3)
# Table 7 recovery
T7 = {'attenuated': ('0.612', '0.672', '0.678'), 'costly': ('0.690', '0.765', '0.785'), 'access': ('0.681', '0.756', '0.812'),
      'access_x_access': ('0.630', '0.711', '0.762'), 'costly_x_costly': ('0.700', '0.762', '0.786')}
for g, (a, l, o) in T7.items():
    chk(f'T7 acts {g}', a, rec.loc[(g, 'acts'), 'correct_family'], 3); chk(f'T7 labels {g}', l, rec.loc[(g, 'labels_eta0.25'), 'correct_family'], 3)
    chk(f'T7 oracle {g}', o, rec.loc[(g, 'oracle'), 'correct_family'], 3); chk(f'T7 null {g}', a, rec.loc[(g, 'null_labels'), 'correct_family'], 3)
SG = {'attenuated': ('0.053', '0.098', '0.54'), 'costly': ('0.068', '0.127', '0.54'), 'access': ('0.050', '0.084', '0.59'),
      'access_x_access': ('0.044', '0.073', '0.61'), 'costly_x_costly': ('0.085', '0.161', '0.53')}
for g, (l, o, r) in SG.items():
    chk(f'T7 sig gain labels {g}', l, pg.loc[(g, 'labels_eta0.25'), 'sig_gain'], 3); chk(f'T7 sig gain oracle {g}', o, pg.loc[(g, 'oracle'), 'sig_gain'], 3)
    chk(f'T7 ratio {g}', r, pg.loc[(g, 'labels_eta0.25'), 'sig_ratio'], 2)
# power (dialogues per group for 80 % power, A on own side)
for (m, pair, v) in [('A_dbelief', 'attenuated vs costly', 17), ('B_pW_low', 'attenuated vs costly', 100), ('A_hidden_shift', 'attenuated vs costly', 17), ('nonc_A', 'attenuated vs costly', 200),
                     ('restrict_A', 'attenuated vs access', 12), ('B_pW_low', 'attenuated vs access', 70), ('restrict_A', 'costly vs access', 12), ('B_pC_high', 'costly vs access', 12)]:
    chk(f'n80 {m} {pair}', str(v), N80[m][pair], 0)
assert N80['A_dbelief']['costly vs access'] is None, 'dbelief separates cost vs access?'
# v3.5 additions: all-world non-concession contrast, own-side label gains (Table 6 is all worlds), number of one-factor label-gain checks
n80 = pd.read_csv(R / 'H3_power_n80.csv').set_index(['split', 'pair', 'channel'])
chk('n80 nonc_A attenuated vs costly, all worlds', '25', n80.loc[('all', 'attenuated vs costly', 'nonc_A'), 'n_80'], 0)
rec_own = pd.read_csv(R / 'H2_recovery.csv'); rec_own = rec_own[rec_own.split == 'A_own0==1'].set_index(['generator', 'channel'])
for g, v in {'attenuated': '2', 'costly': '11', 'access': '5'}.items():
    chk(f'own-side label gain (points) {g}', v, 100 * (rec_own.loc[(g, 'labels_eta0.25'), 'correct_family'] - rec_own.loc[(g, 'acts'), 'correct_family']), 0)
chk('one-factor label-gain checks', '24', len(pd.read_csv(R / 'H4_label_gain_corners.csv')), 0)
# Part I
chk('engine correct', '129', sc.loc['engine', 'strict_correct'], 0); chk('engine macroF1', '0.757', sc.loc['engine', 'macro_f1'], 3)
chk('prompt correct', '145', sc.loc['direct_prompt', 'strict_correct'], 0); chk('prompt macroF1', '0.837', sc.loc['direct_prompt', 'macro_f1'], 3)
chk('keyword acc', '0.25', kb.loc['keyword_rules', 'strict_acc'], 2); chk('keyword macroF1', '0.195', kb.loc['keyword_rules', 'macro_f1'], 3)
chk('keyword correct', '43', round(kb.loc['keyword_rules', 'strict_acc'] * 169), 0); chk('keyword nonshift', '43', round(kb.loc['keyword_rules', 'non_shift_acc'] * 149), 0)
chk('majority correct', '20', round(kb.loc['majority_class(BUILD)', 'strict_acc'] * 169), 0); chk('random correct', '17', round(kb.loc['uniform_random(seed0)', 'strict_acc'] * 169), 0)
chk('random macroF1', '0.099', kb.loc['uniform_random(seed0)', 'macro_f1'], 3); chk('majority macroF1', '0.021', kb.loc['majority_class(BUILD)', 'macro_f1'], 3)
chk('tierB near dups', '15', bc.loc['B', 'near_dup_pairs_J5_ge_0_3'], 0); chk('tierA words max', '71', bc.loc['A', 'words_max'], 0); chk('tierB words max', '47', bc.loc['B', 'words_max'], 0)

df = pd.DataFrame(rows); (ROOT / 'results').mkdir(exist_ok=True); df.to_csv(ROOT / 'results' / 'manuscript_number_check.csv', index=False)
bad = df[~df.match | ~df.in_tex]
print(f'{len(df)} numbers checked, {int((~df.match).sum())} mismatches, {int((~df.in_tex).sum())} not found in tex')
if len(bad): print(bad.to_string(index=False))
sys.exit(1 if len(bad) else 0)
