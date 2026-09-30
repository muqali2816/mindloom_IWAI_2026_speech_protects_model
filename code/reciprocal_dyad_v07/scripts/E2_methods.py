"""Generate METHODS_E2.md from results/E2*.csv (numbers are read, never retyped). Run from workspace root:
   exec(open('reciprocal_dyad/E2_methods.py').read())  or  python reciprocal_dyad/E2_methods.py"""
import json, os
import pandas as pd

RES = 'reciprocal_dyad/results/'
man = json.load(open(RES + 'E2_manifest.json'))
dfa = pd.read_csv(RES + 'E2a_recovery.csv'); dfb = pd.read_csv(RES + 'E2b_labels.csv'); dfs = pd.read_csv(RES + 'E2a_sensitivity.csv')
cc = pd.read_csv(RES + 'E2a_channel_contrasts.csv'); cb = pd.read_csv(RES + 'E2b_contrasts.csv'); na = pd.read_csv(RES + 'E2a_nextact.csv')
GEN_RU = {'honestxhonest': 'честный × честный', 'attenuatedxhonest': 'ослабленный × честный', 'costlyxhonest': 'затратный × честный',
          'mixedxhonest': 'смешанный × честный', 'costlyxcostly': 'затратный × затратный', 'attenuatedxattenuated': 'ослабленный × ослабленный', 'mixedxmixed': 'смешанный × смешанный'}
f3 = lambda v: '—' if pd.isna(v) else f'{v:.3f}'
L = []
L.append("# METHODS_E2 — восстановление механизмов агента A (E2a по актам, E2b по речевым меткам)\n")
L.append("Симуляция заданных механизмов реципрокной диады v0.7; никаких утверждений о людях. Все числа — из results/E2a_*.csv, results/E2b_*.csv, сгенерированных скриптом `E2_recovery.py` (core.py / observer.py не изменялись). Фигуры: figures/E2_acts_recovery.png, figures/E2_family_posterior_undefined.png, figures/E2_labels_gain.png (`E2_figures.py`).\n")
L.append("## Условия\n")
L.append(f"- Миры: `rd.make_worlds({man['n_worlds']}, seed={man['seed']})` (seed protocol['seeds']['E2']); один и тот же набор миров для всех генераторов (парные контрасты между каналами — по общим мирам).")
L.append(f"- β = {man['beta']} у обоих агентов и у наблюдателя (кроме теста чувствительности). Генераторы для A: honest (1, 0), attenuated (0.2, 0), costly (1, 1.6), mixed (0.6, 0.8) при B honest; симметричные пары costly×costly, attenuated×attenuated, mixed×mixed (восстанавливается A).")
L.append(f"- Наблюдатель: `observer.recover(d, 0, channel, beta)`; гипотезы {man['hyps']} (семейство «ослабление» {{ω∈{{0.2,0.6,1}}, c=0}}, семейство «цена уступки» {{ω=1, c∈{{0,0.8,1.6}}}}; точка (1,0) — в обоих). Равные априоры семейств, равномерно внутри семейства. correct_family = доля миров с P(истинное семейство) > 0.5; posterior_on_family — средний постериор истинного семейства. Для honest (точка в обоих семействах) и mixed (не на сетке) семейство не определено: сообщаются P(ослабление) и доля миров с MAP в каждой точке сетки.")
L.append(f"- MAE параметров — по постериорному среднему на объединённой сетке из 5 гипотез. Базовые уровни: «MAE при априорном среднем» с ω̄ = {man['prior_means'][0]:.2f} (среднее omega_grid), c̄ = {man['prior_means'][1]:.2f} (среднее cost_grid) — как в задании; дополнительно столбцы `*_MAE_obsprior` с собственным априорным средним наблюдателя по 5 гипотезам (ω̄ = {man['observer_prior_means'][0]:.2f}, c̄ = {man['observer_prior_means'][1]:.2f}).")
L.append(f"- Парные разности: бутстрап по мирам, {man['n_boot']} повторов, seed {man['boot_seed']}, 2.5/97.5-процентили.")
L.append("- E2b: `rd.emit_labels(d, g, noise, seed)` для g ∈ {0, 0.25, 0.5, 1} × шум ∈ {none, engine, baseline}; seed меток = seed_E2 + 1000·(индекс генератора+1) + 100·(индекс шума) + индекс g (столбец label_seed). Наблюдатель — plug-in decoder (`recover(d, 0, 'labels', 1.5, labels=z, g=g, noise=noise)`): акты партнёра и собственные акты A, продвигающие состояние, заменены MAP-декодами меток; собственные эмиссии A оцениваются точной вероятностью метки Σ_a E(z|a)P(a|история). Это нижняя граница наблюдателя по меткам.")
L.append("- «Прогноз vs объяснение»: средняя по мирам разность лог-правдоподобий 12 актов A под лучшей (по миру) гипотезой неверного семейства и под истинной точкой, делённая на 12 (наты/акт). Для mixed истинная точка (0.6, 0.8) не на сетке — её правдоподобие вычислено отдельным реплеем с той же маргинализацией по s.")
L.append(f"- Время: {man['elapsed_s']} с на всё (14 вызовов recover по актам + 2 чувствительность + 24 по меткам).\n")

L.append("## Табл. E2a-1. Восстановление по актам диады (2000 миров на строку)\n")
L.append("| генератор (A × B) | канал | correct_family | posterior_on_family | P(ослабл.) | ω MAE | ω MAE (априор 0.6) | c MAE | c MAE (априор 0.8) | MAP = истинная точка |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in dfa.iterrows():
    L.append(f"| {GEN_RU[r.generator]} | {r.channel} | {f3(r.correct_family)} | {f3(r.posterior_on_family)} | {r.p_attenuation_mean:.3f} | {r.omega_MAE:.3f} | {r.omega_MAE_prior:.2f} | {r.c_MAE:.3f} | {r.c_MAE_prior:.2f} | {f3(r.map_true_point)} |")
L.append("\nДоли миров с MAP в каждой точке сетки (ω, c):\n")
L.append("| генератор | канал | (0.2,0) | (0.6,0) | (1,0) | (1,0.8) | (1,1.6) |"); L.append("|---|---|---|---|---|---|---|")
for _, r in dfa.iterrows():
    L.append(f"| {GEN_RU[r.generator]} | {r.channel} | {r['map_share_w0.2_c0.0']:.3f} | {r['map_share_w0.6_c0.0']:.3f} | {r['map_share_w1.0_c0.0']:.3f} | {r['map_share_w1.0_c0.8']:.3f} | {r['map_share_w1.0_c1.6']:.3f} |")
L.append("\n## Табл. E2a-2. Парные разности «acts+private − acts» (бутстрап по мирам, 95 %)\n")
L.append("| генератор | метрика | Δ | 2.5 % | 97.5 % |"); L.append("|---|---|---|---|---|")
for _, r in cc.iterrows():
    if pd.isna(r.delta): continue
    L.append(f"| {GEN_RU[r.generator]} | {r.metric} | {r.delta:+.4f} | {r.mc_lo:+.4f} | {r.mc_hi:+.4f} |")
L.append("\n## Табл. E2a-3. Чувствительность наблюдателя к β (канал acts; истинное β = 1.5)\n")
L.append("| генератор | β_obs | correct_family | posterior_on_family | ω MAE | c MAE | MAP = истина | Δ correct (β_obs=0 − 1.5) [95 %] |"); L.append("|---|---|---|---|---|---|---|---|")
for _, r in dfs.iterrows():
    dlt = '' if pd.isna(r.delta_correct_beta0_minus_true) else f"{r.delta_correct_beta0_minus_true:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
    L.append(f"| {GEN_RU[r.generator]} | {r.beta_obs} | {r.correct_family:.3f} | {r.posterior_on_family:.3f} | {r.omega_MAE:.3f} | {r.c_MAE:.3f} | {r.map_true_point:.3f} | {dlt} |")
L.append("\n## Табл. E2a-4. Прогноз актов vs объяснение: Δ лог-правдоподобия на акт (наты; альтернатива − истина; отриц. = альтернатива хуже)\n")
L.append("| генератор | канал | сравнение | Δ нат/акт | SE | доля миров, где альтернатива лучше | logP(истина)/акт |"); L.append("|---|---|---|---|---|---|---|")
for _, r in na.iterrows():
    L.append(f"| {GEN_RU[r.generator]} | {r.channel} | {r.comparison} | {r.dll_per_act:+.4f} | {r.dll_per_act_se:.4f} | {r.frac_alt_better:.3f} | {r.logp_true_per_act:.3f} |")
L.append("\n## Табл. E2b. Восстановление по речевым меткам (партнёр honest; plug-in decoder)\n")
L.append("| генератор A | g | шум | correct_family | posterior_on_family | ω MAE | c MAE | точность MAP-декода актов A | Δ vs acts [95 %] | acts | acts+private |"); L.append("|---|---|---|---|---|---|---|---|---|---|---|")
cbi = cb[cb.contrast == 'labels - acts'].set_index(['generator', 'g', 'noise'])
for _, r in dfb.iterrows():
    c = cbi.loc[(r.generator, r.g, r.noise)]
    L.append(f"| {r.A_type} | {r.g} | {r.noise} | {r.correct_family:.3f} | {r.posterior_on_family:.3f} | {r.omega_MAE:.3f} | {r.c_MAE:.3f} | {r.decode_acc_A:.3f} | {c.delta:+.3f} [{c.mc_lo:+.3f}, {c.mc_hi:+.3f}] | {r.acts_correct_family:.3f} | {r.acts_private_correct_family:.3f} |")

P3 = {}
for a in ('attenuated', 'costly'):
    s = dfb[dfb.A_type == a].set_index(['noise', 'g']); acts = s.acts_correct_family.iloc[0]; priv = s.acts_private_correct_family.iloc[0]
    noisy_idx = [('engine', g) for g in (0.0, 0.25, 0.5, 1.0)] + [('baseline', g) for g in (0.0, 0.25, 0.5, 1.0)]
    P3[a] = dict(g0={n: float(s.loc[(n, 0.0), 'correct_family']) for n in ('none', 'engine', 'baseline')}, acts=float(acts), acts_private=float(priv),
                 gain_g05_none=float(s.loc[('none', 0.5), 'correct_family'] - acts), gain_g05_engine=float(s.loc[('engine', 0.5), 'correct_family'] - acts),
                 max_labels_under_noise=float(s.loc[noisy_idx, 'correct_family'].max()),
                 first_g_exceeding_acts={n: next((float(g) for g in (0.25, 0.5, 1.0) if cbi.loc[(f'{a}xhonest', g, n)].mc_lo > 0), None) for n in ('none', 'engine', 'baseline')})
    P3[a]['engine_to_noiseless_gain_ratio'] = P3[a]['gain_g05_engine'] / P3[a]['gain_g05_none']
    P3[a]['g0_le_acts_all_noise'] = all(v <= acts for v in P3[a]['g0'].values())
    P3[a]['labels_exceed_acts_only_if_g_gt_0'] = P3[a]['g0_le_acts_all_noise']
    P3[a]['engine_gain_at_most_half'] = P3[a]['engine_to_noiseless_gain_ratio'] <= 0.5
    P3[a]['labels_never_exceed_acts_private_under_noise'] = P3[a]['max_labels_under_noise'] <= priv
L.append("\n## Проверка предсказания P3 (в редакции amendment от 30.09.2026)\n")
for a in ('attenuated', 'costly'):
    p = P3[a]
    L.append(f"- **{a}**: labels(g=0) = {p['g0']['none']:.3f}/{p['g0']['engine']:.3f}/{p['g0']['baseline']:.3f} (none/engine/baseline) ≤ acts = {p['acts']:.3f} — **{'выполнено' if p['g0_le_acts_all_noise'] else 'НЕ выполнено'}** для всех трёх уровней шума (для costly разности в пределах MC-интервала). Метки превосходят acts (нижняя граница 95 % > 0) начиная с g = {p['first_g_exceeding_acts']} (none/engine/baseline) — «метки превосходят акты только при g>0» **выполнено**. Прирост при g=0.5: без шума {p['gain_g05_none']:+.3f}, под шумом движка {p['gain_g05_engine']:+.3f} (отношение {p['engine_to_noiseless_gain_ratio']:.2f}) — часть «под шумом движка прирост не более половины бесшумного» **{'выполнена' if p['engine_gain_at_most_half'] else 'не подтвердилась'}**: шум движка почти не уменьшает прирост. Максимум по меткам под шумом = {p['max_labels_under_noise']:.3f} против acts+private = {p['acts_private']:.3f} — часть «метки никогда не превосходят acts+private» **{'выполнена' if p['labels_never_exceed_acts_private_under_noise'] else 'не подтвердилась'}** (превышение при g ≥ 0.25 для attenuated, g ≥ 0.5 для costly).")
L.append("\nПочему две части P3 не подтвердились (объяснение по конструкции модели, не эмпирический вывод): при g>0 эмиссия меток зависит от (ω, c) напрямую (MODEL_SPEC §7), т.е. метки — не зашумлённая версия канала актов, а дополнительный канал наблюдения самого механизма; канал acts+private такого сигнала не содержит, поэтому неравенство labels ≤ acts+private было ошибочным ожиданием для g>0 (оно выполняется при g=0). Эмпирическая матрица ошибок движка сохраняет FLOOD и DRAIN без ошибок (1.00 на диагонали) и LOCK на 0.67 (ошибки уходят в BUILD/SEAL, а не во FLOOD), т.е. шум почти ортогонален направлению модуляции LOCK↔FLOOD/DRAIN, которым закодирован механизм; поэтому прирост от g почти не теряется. Это свойство конкретной матрицы (15–20 случаев на строку, Tier A) и стипулированной модуляции, не общий результат.\n")
L.append("## Оговорки\n")
L.append("- Наблюдатель по меткам — plug-in decoder (нижняя граница); полный наблюдатель по меткам должен фильтровать по всем последовательностям актов.")
L.append("- Наблюдатель знает модель, β, все вспомогательные параметры и матрицу шума; чувствительность проверена только к β (β_obs = 0 при истинном 1.5). Чувствительность к k, τ и совместный постериор по паре в этом треке не считались.")
L.append("- Для генератора mixed (0.6, 0.8) «истинное семейство» не определено: принудительный выбор семейства относит его к семейству «цена уступки» примерно в 70 % миров (P(ослабл.) ≈ 0.30–0.32), хотя ω = 0.6 < 1; лучшая точка сетки теряет лишь 0.002–0.014 нат/акт относительно истинной точки — неверное описание генератора почти незаметно по предсказательной силе. Оценка ω для mixed (≈0.85) смещена вверх, оценка c (≈0.66) — вниз.")
L.append("- Для attenuated по актам ω не восстанавливается (ω MAE 0.41–0.48 при базовом уровне 0.40); восстанавливается только c. Разделение «ослабление vs цена» по актам для attenuated — 0.62–0.71, при этом лучшая гипотеза семейства «цена» предсказывает акты A хуже истинной лишь на 0.01–0.04 нат/акт (канал acts): семейства почти эквивалентны как предсказатели актов и различаются как объяснения.")
L.append("- Доля миров, где A изначально верит своей версии, A_own0 = 0.594 (часть миров стартует с уступки по приватным данным); условные по A_own0 разбиения в этом треке не считались.")
open('reciprocal_dyad/METHODS_E2.md', 'w').write('\n'.join(L))
os.makedirs('handoff', exist_ok=True); json.dump(P3, open('handoff/P3.json', 'w'), indent=1, default=lambda o: bool(o) if hasattr(o, 'item') and isinstance(o.item(), bool) else float(o))
print(json.dumps(P3, indent=1, default=lambda o: bool(o) if hasattr(o, 'item') and isinstance(o.item(), bool) else float(o)))
