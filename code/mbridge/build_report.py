"""Russian model proposal, numerical audit, and English manuscript insert."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as PlotPath
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';D=ROOT/'report';D.mkdir(exist_ok=True)
F='/usr/share/fonts/truetype/dejavu/'
for n,f in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(n,F+f))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DVB',italic='DV',boldItalic='DVB')
styles=getSampleStyleSheet()
for name,size,lead,weight,color in [('B',9.4,13.5,'DV','#182c35'),('S',8.0,11,'DV','#233d49'),('H',13.3,17,'DVB','#173f51'),('T',20,25,'DVB','#173f51'),('C',8.1,11,'DV','#182c35')]:
    styles.add(ParagraphStyle(name=name,fontName=weight,fontSize=size,leading=lead,spaceAfter=8 if name!='C' else 0,textColor=colors.HexColor(color)))
story=[]
def p(t,small=False):story.append(Paragraph(t,styles['S' if small else 'B']))
def h(t):story.append(Paragraph(t,styles['H']))
def page():story.append(PageBreak())
def table(rows,widths):
    t=Table([[Paragraph(str(v),styles['C']) for v in rr] for rr in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f1f3')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#53727e')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#a0b4bc')),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,11)])

# Exact model diagram; all arrows correspond to implemented dependencies.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.5,4.0));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
boxes={'A':(1.0,4.0,2.7,.7,'Agent A: belief + policy'), 'B':(5.1,4.0,2.7,.7,'Agent B: belief + policy'),
       'F':(2.7,2.8,3.4,.75,'Acts + invitation / restriction'),
       'L':(2.7,1.6,3.4,.7,'Access to joint verification'),
       'E':(2.7,.4,3.4,.7,'Fresh shared evidence'),
       'Z':(7.0,2.1,2.7,.9,'Noisy speech labels\nExternal observer')}
for name,(x,y,w,hh,label) in boxes.items():
    ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=.07',facecolor='#edf4f5' if name!='Z' else '#fff0dc',edgecolor='#315f6c',linewidth=1.2))
    ax.text(x+w/2,y+hh/2,label,ha='center',va='center',fontsize=9)
def arr(a,b,curve=0):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#315f6c',connectionstyle=f'arc3,rad={curve}',linewidth=1.2))
arr((2.35,4),(3.8,3.6));arr((6.45,4),(5.2,3.6));arr((4.4,2.8),(4.4,2.3));arr((4.4,1.6),(4.4,1.1));arr((6.1,3.15),(7,2.7))
arr((2.7,.75),(1.0,4.25),-.45)
route=PlotPath([(6.1,.75),(6.55,.75),(6.55,3.85),(6.45,4.0)],[PlotPath.MOVETO,PlotPath.LINETO,PlotPath.LINETO,PlotPath.LINETO])
ax.add_patch(FancyArrowPatch(path=route,arrowstyle='-|>',mutation_scale=12,color='#315f6c',linewidth=1.2))
fig.tight_layout(pad=.2);fig.savefig(D/'dyadic_bridge.png',dpi=220);fig.savefig(D/'dyadic_bridge.pdf');plt.close(fig)

story.append(Paragraph('Речь управляет возможностью учиться',styles['T']))
p('<b>Предложение связки двух частей Speech that Protects a Model</b><br/>Рабочий прототип и результаты · 30 сентября 2026')
p('<b>Основной вопрос:</b> может ли взаимодействие сохранять исходные ожидания потому, что участники ограничивают совместную проверку, хотя способны нормально обновлять убеждения по полученным свидетельствам? И позволяет ли речевая разметка предсказывать это ограничение сверх обычных кодов действий?')
h('1. Что именно связывает две части')
p('Часть I описывает наблюдаемую организацию обмена: запрос, разрешение ответить, ограничение допустимого ответа, закрытие темы. Часть II задаёт процесс, в котором такие различия меняют поступление следующего свидетельства и поэтому последующее обучение. Метки становятся измерительным каналом над речевыми действиями с определёнными последствиями.')
table([['Одна исходная позиция','Разная организация продолжения'],
['«Я уверен, что расчёт верный. Покажи, где расхождение».','Утверждение позиции + приглашение к проверке.'],
['«Я уверен, что расчёт верный. Возражения не обсуждаем».','Утверждение позиции + ограничение проверки.']], [260,239])
p('Это сконструированный пример. Из второй реплики нельзя установить мотив, неизменность частного убеждения или психологическую защиту. Однако можно отдельно проверить, ограничивает ли такая форма продолжение обмена. Именно это различие теряется в одном коде ASSERT.')
h('Что поправлено относительно варианта A Claude')
p('• Шумные метки, зависящие только от действий, могут терять информацию. При условной независимости меток от механизма <b>добавление</b> их к уже наблюдаемым действиям должно давать нулевой выигрыш. Метки сами по себе не обязаны равняться действиям.<br/>• Не задаём прямую эмиссию «больше c → больше LOCK» как доказательство механизма. Сначала моделируем выбор ограничения и его последствия; метка зависит от выбранной формы.<br/>• κ пилота не превращаем в вероятность правильной классификации.<br/>• Не объединяем конфигурации, SEAL и SHIFT в один набор скрытых режимов. Прототип реализует только две ограниченные формы.')
p('<b>Статус:</b> формальная связка и исследовательский тест в симуляции уже реализованы. Это ещё не эмпирическая валидация речевой диагностики. Исходная статья не изменена.',True)

page();h('2. Два участника и два канала наблюдения')
story.append(Image(str(D/'dyadic_bridge.png'),width=490,height=261))
p('У каждого участника есть собственное убеждение о бинарном факте. Начальные вероятности противоположны: 0,75 и 0,25. Оба выбирают сочетание обычного действия u и формы f: приглашение к проверке или ограничение проверки. Действия обоих влияют на общий канал свежих свидетельств; каждое новое свидетельство учитывается один раз.')
table([['Уровень','Что в нём находится','Статус'],
['Участники','Частные убеждения qᴬ, qᴮ; чувствительность ω; цена уступки c; цена публичного опровержения γ.','Скрыты от внешнего наблюдателя; параметры заданы генератором.'],
['Взаимодействие','Действия uᴬ, uᴮ и формы fᴬ, fᴮ.','Участники видят предыдущий ход партнёра. Наблюдатель видит действия, но форму — через шумные метки.'],
['Среда','Факт x, доступ к проверке, свежий сигнал e или отсутствие сигнала.','Правило перехода задано. Запись e доступна наблюдателю в основных сравнениях.'],
['Разметка','Вероятность выхода z при выбранной форме f.','Две строки архивной матрицы ошибок с явными допущениями о переносе.']], [91,230,178])
p('Таким образом, различаются два наблюдателя: сам партнёр воспринимает выбранную форму; исследователь получает её несовершенную автоматическую разметку. В тексте статьи нельзя смешивать эти уровни.',True)
p('Прогноз участника минимален: при выборе следующего действия он предполагает, что партнёр сохранит предыдущие действие и форму. Это полноценное взаимное влияние двух действующих агентов, но без рекурсивной теории психики, обучения модели партнёра или решения равновесия игры.',True)

page();h('3. Явная генеративная модель')
p('<b>Доступ.</b> Пусть f = 1 означает приглашение к проверке, f = 0 — ограничение. Если кто-то запросил информацию, базовая вероятность её появления равна 0,90; иначе 0,35. Совместное ограничение действует по формуле:')
p('<b>λ<sub>t+1</sub> = 0,05 + (λ<sub>base</sub>(uᴬ<sub>t</sub>, uᴮ<sub>t</sub>) − 0,05) fᴬ<sub>t</sub> fᴮ<sub>t</sub>.</b>')
p('Это заданное причинное допущение: для полной совместной проверки нужна открытость обоих. Даже при ограничении остаётся вероятность 0,05. Ограничение не является поглощающим состоянием и не гарантирует отсутствия обучения.')
p('<b>Свидетельства и обновление.</b> Надёжность сигнала ρ = 0,75. Чувствительность меняет только его направление: ρ<sub>ω</sub> = ρ<super>ω</super> / [ρ<super>ω</super> + (1 − ρ)<super>ω</super>]. Частота поступления сохраняется. При наблюдении сигнала выполняется точное байесовское обновление под этой субъективной моделью; отсутствие сигнала о факте неинформативно.')
p('<b>Выбор действия и формы.</b> Для каждого сочетания (u,f) участник вычисляет ожидаемую стоимость и информационный выигрыш:')
p('<b>Gᵢ(u,f) = Rᵢ(u,qᵢ,cᵢ) + δ(1−f) + γᵢ λ̂ᵢ(u,f) dᵢ − λ̂ᵢ(u,f) I<sub>bin</sub>(qᵢ,ρ<sub>ωᵢ</sub>).</b><br/><b>Pᵢ(u,f) = exp[−Gᵢ(u,f)/τ] / Σ<sub>v,h</sub> exp[−Gᵢ(v,h)/τ].</b>')
p('R — прежняя стоимость утверждения, уступки и других действий. δ = 0,12 — условная стоимость ограничения. d — субъективная вероятность противоречащего исходной публичной позиции сигнала при его поступлении. γ оценивает нежелательность такого публичного исхода; это <b>новый параметр</b>, не цена уступки c и не физиологическая нагрузка. λ̂ использует предыдущий ход партнёра. τ = 0,20.')
p('<b>Измерительный канал.</b> P(zᵢ<sub>t</sub> | fᵢ<sub>t</sub>) = C<sub>f,z</sub>. При известной форме в эмиссии нет прямой зависимости от ω, c, γ или частного убеждения. Для последующей работы над естественным текстом вместо этого нужна оценка P(z | f, контекст) на независимых данных.')
p('<b>Связь с active inference.</b> Действие изменяет будущую доступность наблюдений, которая входит в расчёт ожидаемого информационного выигрыша. Здесь явно заданы исходные убеждения, наблюдения, переходы, предпочтения и вероятностная политика. Используется ограниченная форма «ожидаемая стоимость минус информация»; один этот результат не отличает её от всех моделей ожидаемой полезности.')

page();h('4. Что показала первая симуляция')
p('По 2 000 диад в каждом условии, общие случайные числа, 12 обменов (24 индивидуальных решения). Это новая задача: её числа нельзя добавлять строкой к прежней таблице 5, где были один агент и другое сравнение. Параметры, все условия и результаты сохранены в пакете.')
table([['Условие','Число свидетельств','Вероятность истины','Различие убеждений'],
['Исходное: ω=1, c=γ=0','2,701','0,708','0,293'],
['Ослабление: ω=0,1','2,497','0,525','0,496'],
['Дорогая уступка: c=1,4','2,993','0,712','0,286'],
['Публичное опровержение дорого обоим: γ=2','1,821','0,658','0,350'],
['γ=2 только одному','2,057','0,670','0,336'],
['γ=2 обоим; с 7-го обмена проверка доступна независимо от ограничения','6,440','0,855','0,146']], [227,87,92,93])
p('В условии γ = 2 чувствительность к свидетельствам остаётся нормальной: ω = 1. Обучение слабее потому, что агентам поступает меньше свидетельств. Разница в вероятности истины относительно исходного условия равна −0,0496 [−0,0568; −0,0424], в числе свидетельств −0,8805 [−0,9259; −0,8351]. Интервалы — парные 95% интервалы Монте-Карло.')
p('Снятие γ с 7-го обмена повышает вероятность истины на 0,0222 [0,0172; 0,0271]. Принудительное открытие только одной стороны — на 0,0113 [0,0065; 0,0161]. Обход ограничения для совместной проверки — на 0,1968 [0,1872; 0,2064]. Последний эффект одновременно отражает повышение доступности до 0,90; его нельзя толковать как эффект одних слов при неизменном поступлении данных.')
p('<b>Существенная граница:</b> при γ=2 и отключённом информационном выигрыше вероятность истины равна 0,655 вместо 0,658. Основная потеря обучения сохраняется. Active inference здесь задаёт явную проверяемую модель, но пока не демонстрирует уникального объяснительного преимущества.')
p('Частичное сохранение исходных убеждений в конечной серии не доказывает устойчивый аттрактор, гистерезис или человеческий защитный режим. В прототип не встроены именованные скрытые фазы LOCK/FLOOD и минимальное время пребывания в них.',True)

page();h('5. Как первая часть становится измерением')
p('Из архивных прогнозов восстановлена матрица: 169 случаев, 129 верных основных выходов. Для прототипа взяты только строки SEEK (20 случаев) и LOCK (15). Приглашение моделируется кандидатом на SEEK, явное ограничение допустимого ответа — кандидатом на LOCK. Это ограниченная модельная привязка; остальные конфигурации здесь не проверяются.')
p('<b>Неожиданный риск:</b> в архиве SEEK всегда распознан как SEEK; LOCK распознан как LOCK 10 раз, SEAL 3 раза, BUILD 2 раза. В этих двух строках выходы не пересекаются. Если механически перенести их в мир только с двумя формами, форма будет восстанавливаться идеально даже при ошибочной метке. Такой результат был бы искусственно оптимистичным.')
p('Поэтому основной пример использует сглаживание: по 0,5 псевдонаблюдения на каждый из 11 выходов, затем 25% смеси с общим распределением выходов. Это явно заданный стресс-тест переноса, <b>не измеренная надёжность нового корпуса</b>. Чувствительность к шуму проверена отдельно; неопределённость строк матрицы ещё не интегрирована полностью.')
table([['Генератор','Выбор модели: без / с метками','Улучшение прогноза следующего свидетельства'],
['Исходное условие','54,7% / 67,9%','7,5% меньше log-loss'],
['Ослабление','88,65% / 88,75%','6,9% меньше log-loss'],
['Цена уступки','90,2% / 91,35%','8,4% меньше log-loss'],
['Ограничение доступа','68,45% / 76,65%','8,9% меньше log-loss']], [140,171,188])
p('Оба наблюдателя видят одинаковые действия двух участников и всю уже поступившую последовательность свидетельств. Только один дополнительно видит шумные метки. Фильтр точно суммирует скрытые состояния в заданной модели. Сравниваются четыре известные точки параметров с равными априорами; это не восстановление непрерывных параметров и не диагностика неизвестных человеческих механизмов.')
p('<b>Контроли:</b> если метки зависят лишь от уже видимого действия, добавочный выигрыш равен нулю. Он также исчезает при полностью неинформативном канале меток и при устранении влияния формы на доступ к проверке. Все равенства проверены с точностью 10<super>−12</super>. При усилении дополнительного шума до 50% выигрыш прогноза уменьшается до 0,013–0,017 нат на предсказываемый исход.')
p('Это подтверждает внутреннюю логику связки: добавочная информация поступает о форме взаимодействия и её заданных последствиях. Превосходство полной онтологии над простым детектором «разрешает / ограничивает» здесь не проверено.',True)

page();h('6. Где возможен вклад и как собрать статью')
p('<b>Рабочая формулировка вклада:</b> типизированная речевая разметка рассматривается как несовершенное измерение действий, регулирующих доступ к последующим свидетельствам; её ценность оценивается по добавочному прогнозу развития обмена и по чувствительности к интервенциям в диаде.')
p('Само добавление второго агента не даёт новизны. Friston и Frith уже моделировали взаимное предсказание [1]; Medrano и Sajid — устойчивое расхождение интерпретаций [2]. Tison и Poirier рассматривают речь как изменение возможностей совместного действия [3]. В твоём собственном предшествующем препринте уже заявлены диадные механизмы и ограничение канала свидетельств [4].')
p('Поэтому потенциальная добавка этой работы — <b>конкретная связь измерительной схемы с процессом, её ошибки и проверяемая добавочная прогностическая ценность</b>. Приоритет в литературе не установлен исчерпывающим обзором. Полный текст [4] в этой проверке недоступен; сопоставление с ним основано на опубликованной аннотации.')
h('Предлагаемая логика рукописи')
p('1. Вопрос: как речь меняет возможность пересмотра убеждений?<br/>2. Наблюдения: какие ограничения и приглашения можно кодировать; что известно об ошибках разметки.<br/>3. Базовые альтернативы: слабое усвоение данных и дорогая публичная уступка из прежней модели.<br/>4. Диадное расширение: оба участника изменяют доступ к проверке; отдельный параметр нежелательности публичного опровержения.<br/>5. Измерительный тест: что добавляют метки к действиям и истории данных; отрицательные контроли и интервенции.<br/>6. Границы и эмпирическая программа: проверка формы, последствий, частных убеждений и только затем физиологической цены.')
h('Что должно быть проверено на людях')
p('Сначала независимо разметить приглашение и ограничение в контексте до момента прогноза. Измерить, следует ли за ними доступ к ответу/проверке, и оценить изменения приватных убеждений. Сравнить полную схему с обычными кодами действий, простым бинарным признаком ограничения и текстовым baseline. Разделять выборки по диадам. Рандомизировать доступ к проверке или цену публичного опровержения; отдельно сохранять контроль реальной угрозы и уместных границ. Отсутствие добавочного прогноза должно считаться возможным отрицательным результатом.')
p('Аллостатическая нагрузка остаётся отдельной гипотезой. Стоимости c, γ и δ безразмерны и не измеряют расход энергии организма. В симуляции нет физиологии, накопленного вреда или высшего убеждения о собственной ценности.',True)

page();h('7. English insert: the bridge and its limits')
english=[
('Proposed contribution',
'We ask whether speech annotation can measure actions that regulate access to subsequent evidence, rather than serve as a direct readout of a protective mechanism. A factual assertion can coexist with an invitation to inspect counterevidence or with a restriction on admissible replies. These interactional differences may alter the opportunities for belief revision even when evidence sensitivity is intact. We therefore connect the annotation and modelling components through a noisy observation model of interactional form and a dyadic transition model in which that form changes access to joint verification.'),
('Minimal reciprocal model',
'Two agents maintain beliefs about a static binary fact, with initial probabilities 0.75 and 0.25. At each of twelve exchanges, each selects a coarse act and either an invitation or a restriction. The probability of fresh shared evidence depends on both current forms and whether either agent asks for information. Agents update by Bayes under an arrival-preserving subjective likelihood. Joint act-form policies minimise expected loss minus expected information gain. Alongside the original concession cost, a separate parameter penalises expected public evidence contradicting the initial public position. This parameter is a stipulated preference, not an inferred motive, a maintenance cost or an allostatic measure. Each agent forecasts the partner by holding its previous move fixed; no recursive theory of mind is claimed.'),
('Observation model and prediction',
'The annotation channel depends on the selected interactional form, without a direct dependence on the hidden mechanism parameter. Two rows of the archived primary-output confusion matrix provide an empirical starting point, with explicit smoothing and additional state-independent noise as transport assumptions. An exact external filter observes both agents’ coarse actions and the evidence history; a second filter additionally receives noisy form labels. Evaluation compares next-evidence prediction and discrimination among four specified parameter points on separate simulation worlds. It does not estimate a continuous human mechanism or validate the entire annotation ontology.'),
('Exploratory results',
'Across 2,000 paired simulated dyads per condition, aversion to public contradiction reduced mean evidence exposure from 2.701 to 1.821 observations and mean final truth-weighted belief from 0.708 to 0.658 despite intact evidence sensitivity. Adding noisy form labels reduced next-evidence log loss by 6.9–8.9% across the four generators. The gain vanished when labels depended only on already observed acts, when the label channel became state-independent, and when forms ceased to affect evidence access. These are positive controls within a stipulated process and measurement model, not evidence that human protective speech has been identified. The principal reduction in learning also survived removal of expected information gain, limiting claims of explanatory specificity to active inference.')]
for title,body in english:
    p('<b>'+title+'</b>',True);p(body,True)
(D/'Proposed_bridge_sections_EN.txt').write_text('\n\n'.join(a+'\n'+b for a,b in english)+'\n')

page();h('8. Источники, воспроизводимость и нерешённые вопросы')
refs=[
'[1] Friston, K., & Frith, C. (2015). A Duet for One. Consciousness and Cognition, 36, 390–405. DOI: 10.1016/j.concog.2014.12.003. https://discovery.ucl.ac.uk/id/eprint/1460459/',
'[2] Medrano, J., & Sajid, N. (2024). A Broken Duet: Multistable Dynamics in Dyadic Interactions. Entropy, 26(9), 731. https://arxiv.org/abs/2408.03809',
'[3] Tison, R., & Poirier, P. (2021). Active Inference and Cooperative Communication: An Ecological Alternative to the Alignment View. Frontiers in Psychology, 12, 708780. https://doi.org/10.3389/fpsyg.2021.708780',
'[4] Svet, M., & Perera Molligoda Arachchige, A. S. (2026). Defense as Precision Management: Testing Regime Dynamics in Single-Agent and Dyadic Active-Inference Models of Protective Speech. SSRN. https://doi.org/10.2139/ssrn.7245984 (для настоящего сопоставления проверена аннотация).']
for ref in refs:p(ref,True)
h('Содержимое пакета')
p('dyad_model.py — генератор и внешний фильтр; run_experiments.py — все сравнения и извлечение архивной матрицы; validate_independent.py — независимые проверки; build_report.py — отчёт и схема. Каталог results содержит полные таблицы, индивидуальные предсказания, метаданные и параметры. source содержит использованные архивные прогнозы и неизменённый скрипт их пересчёта.')
p('Запуск: python run_experiments.py, затем python validate_independent.py. Основные зависимости: Python, NumPy, pandas. Отчёт дополнительно использует matplotlib и ReportLab. Основные миры: seed 202609302; миры оценки наблюдателей: 202609303. По 2 000 диад на генератор. Новые анализы исследовательские, выполнены после ревью; внешней пререгистрации нет.')
h('Проверки реализации')
p('Независимый скалярный расчёт вероятностей действий совпадает с векторным до 3,33 × 10<super>−16</super>. Полное перечисление коротких скрытых траекторий совпадает с фильтром до 5,56 × 10<super>−16</super>. Проверены нормировка, влияние партнёра в обоих направлениях, отсутствие добавочной информации у условно независимых меток и исчезновение выигрыша при отключении связи формы с доступом.')
h('Чего этот прототип пока не устанавливает')
p('• Реальное причинное влияние конкретной речевой формы на ответы партнёра: здесь оно задано.<br/>• Перенос архивной матрицы на естественные диады и преимущество сложной онтологии перед двумя простыми признаками.<br/>• Устойчивые режимы, аттракторы, переходы SHIFT или гистерезис.<br/>• Восстановление скрытых непрерывных параметров, мотивов, физиологии или самооценки.<br/>• Уникальность active inference по сравнению с конкурентными моделями выбора.<br/>• Полную новизну относительно всей литературы и полного текста предыдущего препринта.')
p('<b>Решение для статьи:</b> использовать эту модель как отдельное, явно исследовательское расширение с измерительным тестом. Сохранить предыдущие результаты как базовые альтернативы. Не заменять ими человеческую проверку и не представлять построенный наблюдательный канал как уже установленную связь речи с психологической защитой.')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#abc0c7'));c.line(48,42,547,42)
    c.setFont('DV',7.8);c.setFillColor(colors.HexColor('#526c78'))
    c.drawString(48,28,'Mindloom · диадная связка · исследовательский прототип · 30.09.2026');c.drawRightString(547,28,str(doc.page))
doc=SimpleDocTemplate(str(D/'Mindloom_dyadic_bridge_RU_2026-09-30.pdf'),pagesize=(595.276,841.89),leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=56,
    title='Речь управляет возможностью учиться: диадная связка',author='Mindloom Research Program')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Report built.')
