"""Create the checked scientific report from completed result tables."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
import model as M
ROOT=M.ROOT;R=ROOT/'results';OUT=ROOT/'report';OUT.mkdir(exist_ok=True)
NAVY='#18354B';TEAL='#087F83';GRAY='#546673';LIGHT='#ECF3F5';ORANGE='#BD5C30'
FONT='/usr/share/fonts/truetype/dejavu/'
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,FONT+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.1,leading=14.3,textColor=HexColor(NAVY),spaceAfter=8),
 'en':ParagraphStyle('en',fontName='Body',fontSize=9.6,leading=13.0,textColor=HexColor(NAVY),spaceAfter=7),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.3,leading=11.4,textColor=HexColor(GRAY),spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.7,leading=11.7,textColor=HexColor(NAVY)),
 'headcell':ParagraphStyle('headcell',fontName='Bold',fontSize=8.7,leading=11.7,textColor=white),
 'h1':ParagraphStyle('h1',fontName='Bold',fontSize=21,leading=26,textColor=HexColor(NAVY),spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='Bold',fontSize=12.3,leading=16,textColor=HexColor(TEAL),spaceBefore=8,spaceAfter=7),
 'kicker':ParagraphStyle('kicker',fontName='Bold',fontSize=9,leading=13,textColor=HexColor(TEAL),spaceAfter=10),
 'callout':ParagraphStyle('callout',fontName='Bold',fontSize=12,leading=17,textColor=HexColor(NAVY),spaceAfter=10),
}
story=[]
def p(t,style='body'):return Paragraph(t,styles[style])
def add(t,style='body'):story.append(p(t,style))
def heading(n,t):
 if story:story.append(PageBreak())
 add(f'MINDLOOM • HYBRID v0.8 • {n:02d}','kicker');add(t,'h1')
def table(rows,widths):
 cells=[[p(str(c),'headcell' if i==0 else 'cell') for c in row] for i,row in enumerate(rows)]
 t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#F2F6F7'),white]),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#CCD8DC'))]))
 story.extend([t,Spacer(1,10)])
def callout(t):
 tab=Table([[p(t,'callout')]],colWidths=[499]);tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HexColor(LIGHT)),('BOX',(0,0),(-1,-1),.6,HexColor('#C9DCDD')),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([tab,Spacer(1,10)])
def row(a,b,suffix='beta1.0'):
 return dyn[(dyn.A==a)&(dyn.B==b)&(dyn.condition==f'{a}__{b}__{suffix}')].iloc[0]
def f(x,n=3):return f'{x:.{n}f}'
def pct(x):return f'{100*x:.1f}%'
def ci_text(mean,lo,hi,n=3):return f'{mean:+.{n}f} [{lo:+.{n}f}; {hi:+.{n}f}]'
def getoc(metric,mode='labels_evidence',gen='all'):
 return oc[(oc.metric==metric)&(oc.comparison==mode+' over actions_evidence')&(oc.generator==gen)].iloc[0]
def footer(c,doc):
 c.setStrokeColor(HexColor('#CBD8DD'));c.line(48,42,547,42)
 c.setFont('Body',8);c.setFillColor(HexColor(GRAY));c.drawString(48,29,'30.09.2026  •  Симуляционная модель; не проверка на людях')
 c.drawRightString(547,29,str(doc.page))

def figures():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 names=M.TYPE_NAMES;labels=['Базовый','Ослабление','Цена уступки','Защита доступа']
 fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
 for ax,metric,title in zip(axes,['p_truth','evidence_count'],['Вероятность истинной версии','Поступившие проверки / диалог']):
  mat=np.array([[row(a,b)[metric] for b in names] for a in names])
  ax.imshow(mat,cmap='Blues',vmin=.80 if metric=='p_truth' else 1.8,vmax=.92 if metric=='p_truth' else 5.1)
  ax.set_xticks(range(4),labels,rotation=28,ha='right',fontsize=8);ax.set_yticks(range(4),labels,fontsize=8);ax.set_title(title,fontweight='bold',fontsize=10);ax.set_xlabel('Профиль B');ax.set_ylabel('Профиль A')
  for i in range(4):
   for j in range(4):ax.text(j,i,f(mat[i,j],3 if metric=='p_truth' else 2),ha='center',va='center',color='white' if mat[i,j]>(.87 if metric=='p_truth' else 3.8) else NAVY,fontsize=10)
 fig.savefig(OUT/'mixed_pairs.png',dpi=220);plt.close(fig)
 tr=pd.read_csv(R/'trajectories.csv');fig,ax=plt.subplots(figsize=(8.7,2.6),layout='constrained')
 for typ,label,color in zip(names,labels,[TEAL,'#767B8E',NAVY,ORANGE]):
  t=tr[tr.condition==f'{typ}__{typ}__beta1.0'];ax.plot(t.event,t.p_truth,label=label,color=color,lw=2)
 ax.set(xlabel='Завершённые ходы (A, B по очереди)',ylabel='Средняя вероятность\nистинной версии',ylim=(.65,.95),xticks=range(0,17,2));ax.grid(axis='y',alpha=.18);ax.legend(ncol=2,fontsize=8,loc='lower right')
 fig.savefig(OUT/'belief_trajectories.png',dpi=220);plt.close(fig)

# Final data only: report must not silently accept a partially completed run.
assert (R/'observer_manifest.json').exists(), 'Observer including sensitivity must be complete'
dyn=pd.read_csv(R/'dynamics.csv');dyn=dyn[dyn.split=='all']
ov=pd.read_csv(R/'observer_overall.csv').set_index('mode');oc=pd.read_csv(R/'observer_contrasts.csv')
ps=pd.read_csv(R/'particle_sensitivity_summary.csv');val=json.loads((R/'independent_validation.json').read_text())
ann=json.loads((R/'annotation_posterior.json').read_text());manifest=json.loads((R/'observer_manifest.json').read_text())
figures()
ref=row('reference','reference');att=row('attenuated','attenuated');cost=row('concession_cost','concession_cost');gam=row('access_protection','access_protection')
arr=getoc('arrival_logloss');act=getoc('act_logloss');cls=getoc('correct_A')

heading(1,'Гибрид собран.\nЧто именно мы проверяем?')
add('Речь, которая меняет доступ к проверке','h2')
callout('Одинаковое «я не меняю позицию» может скрывать разные процессы: слабый учёт данных, дорогую публичную уступку или ограничение доступа к новой проверке.')
add('Гибрид соединяет две части статьи через <b>наблюдаемую форму взаимодействия</b>. Агент выбирает, что сказать и приглашает ли он к проверке. Это влияет на поступление новых данных. Речевые метки служат зашумлённым измерением этой формы; по последовательности актов, меток и проверок внешний наблюдатель делает прогноз.')
add('Что объединено','h2')
table([['Из M-bridge','Из reciprocal v0.7'],['Совместный выбор «акт × форма»; форма меняет доступ к проверке; отдельный параметр γ; нормированное байесовское обновление.','Разные параметры у A и B; разные приватные сведения; речь партнёра как источник информации; вывод о его параметрах; временное REASSURE.'],['Метки зависят от формы поведения.','Каждый участник учитывает предполагаемый ответ другого.']], [249.5,249.5])
add('Три уровня результата','h2')
add('<b>Реализовано:</b> замкнутый диалог двух агентов, явный канал меток, внешний наблюдатель, вмешательства и воспроизводимые расчёты.')
add('<b>Проверено в симуляции:</b> 2 000 общих миров для динамики; 16 сочетаний профилей; 96 отдельных диалогов для наблюдателя; 13 независимых вычислительных и причинных проверок.')
add('<b>Ещё не установлено:</b> что эти параметры описывают защиту у людей; что метки восстанавливают психологический механизм; что удержание защиты создаёт аллостатическую нагрузку.')
add('Это новый исследовательский прототип v0.8. В нём изменены задача, начальные данные и порядок ходов, поэтому его числа нельзя напрямую сравнивать с прежней Таблицей 5. Исходная статья здесь не перезаписана.','small')

heading(2,'Один разговор — по шагам')
add('Есть неизвестный факт с двумя версиями, 0 и 1. A сначала предпочитает 1, B — 0. Каждый получает две собственные зашумлённые записи. Затем идут 8 раундов: 16 поочерёдных ходов.')
add('На ходе говорящий выбирает <b>акт</b> — утверждать 0, утверждать 1, спросить, молчать или предложить безопасную уступку (REASSURE) — и <b>форму</b>: ограничить или пригласить проверку. Слушатель обновляет представление о факте и о партнёре. После этого может поступить новая общая проверка, которую оба учитывают один раз.')
table([['Параметр','Простое значение','Чего он не означает'],['ω — вес проверки','Насколько убедителен новый результат проверки. При ω = 0,3 его субъективная надёжность ниже, чем при ω = 1.','Не общий коэффициент «защитности»; исходные приватные записи не ослабляются.'],['c — цена уступки','Насколько дорого публично сменить текущую позицию. После смены цена относится уже к новой позиции.','Не цена изменения внутреннего убеждения и не физиологическая нагрузка.'],['γ — цена опровержения','Насколько нежелательна будущая проверка, противоречащая позиции после выбранного акта.','Не диагностический признак мотива и не измеренная цена поддержания защиты.']],[96,224,179])
add('Почему это именно диада','h2')
add('Доступ к проверке зависит от текущей формы одного участника и предыдущей формы другого. Каждый оценивает вероятный ответ партнёра. Поэтому действие A меняет условия следующего решения B, а ответ B меняет дальнейшие решения A.')
add('В модели оба участника рассуждают об упрощённом партнёре: предполагают, что тот выбирает только по ожидаемым затратам. Сами они могут учитывать и ценность информации. Это ограниченная модель другого человека, а не бесконечное «я думаю, что он думает…».')
add('REASSURE снимает c у адресата только на следующем ходе. Такое действие заранее задано разработчиком. Его эффективность у реальных людей должна проверяться отдельно.','small')

heading(3,'Явная связь между речью и моделью')
add('Часть I даёт средства разметки. Часть II задаёт процесс, в котором форма взаимодействия имеет последствия. Связь проходит через форму: <b>параметры → выбор акта и формы → доступ к проверке; форма → наблюдаемая метка</b>. Прямого пути «параметр → метка» при известной форме нет.')
table([['Статус','Что именно'],['Наблюдается внешним наблюдателем','Грубые акты обеих сторон; уже поступившие результаты проверки; в расширенном канале — текущая речевая метка до новой проверки.'],['Выводится вероятностно','Профили A и B, скрытые истории форм, приватные исходные сведения. Внутри агента — вероятность факта и профиль партнёра.'],['Задано в этой симуляции','Словарь актов, сетка параметров, форма → вероятность проверки, эффект REASSURE, соответствие restriction/LOCK и invitation/SEEK.'],['Взято из архивных данных','Числа ошибок движка: 15 случаев с основной меткой LOCK и 20 — SEEK. Это лишь две строки прежнего набора из 169 случаев.']],[137,362])
add('Вместо произвольных 25% шума','h2')
add('Для каждой строки канала задаётся распределение Дирихле: архивные числа ошибок плюс 0,5 в каждой ячейке. Это множество возможных матриц ошибок с разными вероятностями. Наблюдатель усредняет по этой неопределённости; одна сгенерированная матрица остаётся постоянной внутри диалога.')
add(f'При равной частоте двух форм медиана информации в метке — <b>{ann["equal_form_prior_MI_nats"][1]:.3f} нат</b>; 95% интервал по матрицам [{ann["equal_form_prior_MI_nats"][0]:.3f}; {ann["equal_form_prior_MI_nats"][2]:.3f}]. «Нат» — единица информации при натуральном логарифме. Это характеристика заданного канала, а не точность диагностики.')
add('Выбор 0,5 тоже является допущением: при добавлении 0,1 и 1,0 медиана информации меняется до 0,669 и 0,384 нат. Это анализ чувствительности канала; прогноз наблюдателя с этими альтернативами не пересчитывался.','small')
add('<b>Ограничение:</b> прежние LOCK и SEEK — более богатые категории, чем один бит «ограничить/пригласить». Их строки использованы как предварительные опорные категории. Нужно собрать отдельную разметку формы в целевых диалогах.')
add('Остальные названия меток могут возникнуть как шум выхода классификатора. Восемь речевых режимов, функция SEAL и переход SHIFT не превращены здесь в отдельные скрытые состояния. Модель пока проверяет узкий мост, а не всю онтологию. Новый текст не генерировался, движок разметки повторно не запускался.','small')

heading(4,'Три механизма дают разные результаты')
add('Ниже — средние по одним и тем же 2 000 мирам при включённой ценности информации. Профили у A и B могут различаться. Внутренняя метрика — средняя вероятность истинной версии у двух агентов, а не процент «правильных людей».')
story.append(Image(str(OUT/'mixed_pairs.png'),width=499,height=177.4));story.append(Spacer(1,10))
table([['Одинаковый профиль у A и B','P(истина)','Проверок','Публично не согласны'],['Базовый',f(ref.p_truth),f(ref.evidence_count,2),pct(ref.public_disagreement)],['Ослабление ω',f(att.p_truth),f(att.evidence_count,2),pct(att.public_disagreement)],['Цена уступки c',f(cost.p_truth),f(cost.evidence_count,2),pct(cost.public_disagreement)],['Защита доступа γ',f(gam.p_truth),f(gam.evidence_count,2),pct(gam.public_disagreement)]],[214,80,80,125])
add('<b>Защита доступа:</b> разность с базовой парой: −1,487 [−1,546; −1,428] проверки и −0,047 [−0,054; −0,040] для P(истина). Здесь ω = 1: данные учитывались бы нормально, но поступают реже. При γ вероятность истины всё же растёт с исходных 0,679 до 0,838: результат означает ослабление обновления.')
add('<b>Цена уступки:</b> публичное несогласие сохраняется у 78,9% пар, хотя P(истина) достигает 0,907. Число проверок даже выше базового. Следовательно, публичная жёсткость сама по себе не доказывает отсутствия внутреннего обновления.')
add('Интервалы — парные 95% интервалы Монте-Карло по мирам, среднее ± 1,96 SE (SE — стандартная ошибка среднего); не интервалы эффектов у людей. В исходно несогласном подмножестве (816 миров, определено до диалога одинаково для всех условий) эффект γ на P(истина) равен −0,080 [−0,093; −0,066].','small')

heading(5,'Что меняют информация и вмешательства')
add('Без информационного члена механизм ограничения доступа сохраняется. Значит, наличие формального active-inference расчёта ещё не доказывает его необходимость для самого эффекта.')
table([['Критерий выбора','Базовая пара:<br/>P(истина) / проверок','Пара γ:<br/>P(истина) / проверок'],['Только ожидаемые затраты (β = 0)',f'{row("reference","reference","beta0.0").p_truth:.3f} / {row("reference","reference","beta0.0").evidence_count:.2f}',f'{row("access_protection","access_protection","beta0.0").p_truth:.3f} / {row("access_protection","access_protection","beta0.0").evidence_count:.2f}'],['Затраты и информация (β = 1)',f'{ref.p_truth:.3f} / {ref.evidence_count:.2f}',f'{gam.p_truth:.3f} / {gam.evidence_count:.2f}']],[219,140,140])
add('Информационный член вычисляет, сколько агент ожидает узнать о факте, приватных сведениях и параметрах партнёра из ближайшей проверки и ответа. Это точная взаимная информация внутри заданной модели. Критерий: <b>ожидаемые затраты − β × информация</b>; предпочтительны меньшие значения. Такая запись эквивалентна полезности с бонусом за информацию [4].')
add('Целевые вмешательства','h2')
table([['Вмешательство','Наблюдаемый эффект в симуляции'],['Пара γ: отменить γ с хода 8','P(истина): 0,838 → 0,871; проверок: 2,159 → 2,951.'],['Пара γ: принудительно приглашать проверку с хода 8','P(истина): 0,838 → 0,891; проверок: 2,159 → 4,198.'],['Пара γ: отключить зависимость доступа от формы на весь диалог','P(истина): 0,916; проверок: 5,692. Это изменение среды и ожидаемого доступа.'],['A с ценой уступки, B базовый: REASSURE против локального sham','Публичное несогласие: 44,85% → 18,75%. P(истина): 0,8884 → 0,8862; улучшения истинности не показано.']],[221,278])
add('Sham отключает только эффект одной навязанной реплики. Более ранние спонтанные REASSURE работают одинаково в обеих ветвях. До вмешательства истории совпадают. Снятие цены уступки может менять публичный ответ без улучшения знаний — это содержательный отрицательный результат.')
add('Для проверки самостоятельной ценности информационного члена нужны предсказания на новых человеческих диалогах и сравнение с альтернативами. Эквивалентные математические записи не различить по одному поведению.','small')

heading(6,'Что дают метки внешнему наблюдателю')
add('96 новых миров: по 24 для каждого профиля A; B во всех сгенерированных диалогах базовый, но наблюдатель этого не знает. Он рассматривает 4 × 4 профиля и все 9 сочетаний приватных исходных данных. Это пилот восстановления на фиксированной сетке.')
table([['Канал','Что доступно'],['Акты + данные','Акты обеих сторон и история уже поступивших проверок; формы скрыты.'],['Акты + данные + метки','То же, плюс зашумлённая метка текущего акта, доступная до новой проверки.'],['Акты + данные + точная форма','Дополнительно истинная форма. Верхний ориентир измерения, недоступный в обычном тексте.']],[184,315])
rows=[['Канал','Ошибка прогноза проверки ↓','Ошибка следующего акта ↓','Профиль A: верно ↑']]
for mode,name in [('actions_evidence','Акты + данные'),('labels_evidence','+ метки'),('forms_evidence','+ точная форма')]:
 r=ov.loc[mode];rows.append([name,f(r.arrival_logloss),f(r.act_logloss),pct(r.correct_A)])
table(rows,[178,112,112,97])
add('Ошибка — средняя отрицательная логарифмическая вероятность наблюдённого исхода (нат); меньше — лучше. Прогноз акта использует только предшествующую историю. Прогноз проверки уже учитывает текущий акт и, если доступна, его метку. Первые четыре хода исключены из средней оценки.')
add(f'<b>Добавление меток:</b> выигрыш прогноза поступления проверки {arr["mean"]:.4f} [{arr.lo:.4f}; {arr.hi:.4f}] нат, или {100*arr["mean"]/ov.loc["actions_evidence","arrival_logloss"]:.1f}% ошибки базового канала. Выигрыш прогноза следующего акта {act["mean"]:+.4f} [{act.lo:+.4f}; {act.hi:+.4f}] нат.')
add(f'<b>Восстановление механизма:</b> изменение точности профиля A {100*cls["mean"]:+.1f} п.п. [{100*cls.lo:+.1f}; {100*cls.hi:+.1f}]. Нельзя переносить успех прогноза проверки на утверждение о распознавании защиты.')
add('Польза меток для проверки ожидаема при заданной причинной связи «форма → доступ». Размер пользы показывает сохранение информации через шумный канал. Он не подтверждает эту причинную связь в человеческом диалоге. Не сравнивать эту задачу с прежними accuracy 0,835 / 0,696.','small')

heading(7,'Что проверено независимо')
add(f'<b>{sum(val["checks"].values())} из {len(val["checks"])} проверок пройдены.</b> Это проверка реализации и внутренних предсказаний модели, а не независимая эмпирическая репликация.')
table([['Проверка','Результат'],['Отдельный скалярный расчёт против векторного ядра','Максимум расхождения политики 2,61 × 10⁻¹⁵; информации 2,00 × 10⁻¹⁵.'],['Байесовское усреднение всех ответов; смена ролей','Ошибки 2,08 × 10⁻¹⁷ и 1,44 × 10⁻¹⁵. Надёжность проверки не зашита как log(3).'],['Текущая публичная позиция','Её смена меняет стоимость и политику: состояние действительно участвует в причинном процессе.'],['Повторные сообщения партнёра','Не создают бесконечный поток независимых приватных записей: информационный вклад ограничен исходными двумя записями.'],['Нулевые контроли','При отключённой связи форма → доступ форма теряет соответствующую причинную роль. Явно неинформативный канал меток не меняет вывод.'],['Короткий полный перебор скрытых форм','576 путей на двух ходах; частицы K = 128 дают ошибку компонента posterior ≤ 0,00221, прогноза ≤ 0,00281.'],['Интегрирование распределения Дирихле','Последовательный расчёт совпал с аналитической формулой; локальное sham не меняет предысторию.']],[196,303])
add('Длинные истории скрытых форм — приближённый расчёт','h2')
add('При известных формах внешний вывод точен на конечной сетке. При скрытых формах основной расчёт хранит K = 8 возможных историй внутри каждого сочетания параметров и приватных данных (144 группы). Posterior — обновлённые вероятности гипотез. Статические группы не выбрасываются. Дополнительно проверены K = 4 и 32 на первых 16 мирах.')
rows=[['Канал','Δ ошибки проверки K32 − K8','Макс. Δ компонента posterior']]
for mode,name in [('actions_evidence','Акты + данные'),('labels_evidence','+ метки')]:
 rr=ps[(ps['mode']==mode)&(ps.paths==32)].iloc[0];rows.append([name,f(rr.mean_delta_from_K8,5),f(rr.max_posterior_component_difference,4)])
table(rows,[207,146,146])
for mode,name in [('actions_evidence','актов'),('labels_evidence','меток')]:
 rr=ps[(ps['mode']==mode)&(ps.paths==32)].iloc[0]
 add(f'На 16 контрольных мирах средняя абсолютная разница ошибки проверки между K = 8 и K = 32 для {name}: {rr.mean_abs_delta_from_K8:.4f} нат. Это численная неопределённость поверх вариации между мирами.','small')
add('Небольшая разница среднего прогноза не гарантирует стабильного решения для каждого отдельного диалога. Оценки восстановления профиля на K = 8 следует считать предварительными. Полные результаты чувствительности и матрицы ошибок включены в архив.','small')

heading(8,'Как теперь связать статью')
add('Начать с конкретного научного вопроса','h2')
add('Работы Friston и Frith рассматривают коммуникацию через взаимное предсказание и обучение общим моделям [1, 2]. Friston, Parr и соавторы моделируют языковой обмен вопросами и ответами [3]. Наш вопрос уже: <b>как различить в наблюдаемом диалоге пересмотр убеждения, сдерживание публичной уступки и ограничение доступа к проверке?</b> Это задача этой статьи, а не доказательство отсутствия других работ о неудачной коммуникации.')
add('Логика разделов','h2')
table([['Раздел','Основная роль'],['Введение и gap','Одна устойчивая реплика совместима с несколькими процессами. Нужны независимые наблюдения и проверяемая связь разметки с процессом.'],['Часть I','Описать признаки и ограничения измерения; отдельно показать результаты движка, прямого LLM и разметчиков.'],['Мост и Часть II','Определить форму и её канал наблюдения; затем диадную модель, динамику и проверку наблюдателя.'],['Обсуждение','Разделить вычислительную корректность, идентифицируемость в симуляции и будущую проверку психологической гипотезы.']],[133,366])
add('Иллюстрация — не экспериментальные данные','h2')
add('<b>Проверка:</b> Alex: «У меня версия A. Сравним исходные файлы?» Sam: «Да. Если в записи другая дата, пересмотрим вывод».<br/><b>Ограничение проверки:</b> Alex: «У меня версия A». Sam: «В файле другая дата». Alex: «Ты опять ищешь ошибки; это не обсуждаем».')
add('Пример показывает доступность проверки, но по этим репликам нельзя установить мотив, внутреннее убеждение или c/γ. Даже отсутствие уступки не равно отсутствию обучения.')
add('Roadmap','h2')
add('<b>1.</b> Отдельно измерять убеждения и публичные ответы. <b>2.</b> Независимо менять надёжность проверки, её доступность и цену публичной уступки. <b>3.</b> Размечать формы по доступному префиксу диалога, проверять на новых диадах. <b>4.</b> Расширить сетку механизмов и проверить ошибки модели партнёра. <b>5.</b> Только затем связывать длительность защитных траекторий с независимыми показателями нагрузки и восстановления.')
add('Аллостатическая гипотеза сохранена как направление проверки. γ и c безразмерны; накопленной физиологической нагрузки, восстановления и соответствующего измерительного канала в v0.8 нет.','small')

# English copy is emitted separately as a plain text manuscript insertion.
en=[]
def enp(text):en.append(text);add(html.escape(text),'en')
heading(9,'English insertion: rationale and model')
enp('A reciprocal observation bridge for model-protective discourse')
enp('Active-inference accounts have described communication through mutual prediction, shared generative models and perceptual learning (Friston & Frith, 2015a,b), and have simulated linguistic question–answer exchange (Friston et al., 2020). We address a narrower measurement problem: persistent public disagreement does not uniquely identify absent belief revision. Reduced uptake of delivered evidence, a cost of changing public commitment, and restricted access to verification can produce different relationships between speech and belief. Our contribution is a computable observation bridge that distinguishes these possibilities in a bounded synthetic dyad. It is not a claim that unsuccessful communication has previously been absent from active-inference research.')
enp('Two agents receive distinct private records about a binary fact and then alternate for sixteen moves. Each chooses a factual statement (0 or 1), ASK, SILENCE or REASSURE, together with an inviting or restrictive interactional form. Verification arrival depends on the current form and the partner’s preceding form. A delivered verification outcome is shared and incorporated once by each participant. The agents have separate parameters θᵢ = (ωᵢ, cᵢ, γᵢ): ω controls the subjective reliability of new verification, c penalizes a change from the current public position, and γ prices expected verification contradicting the position after the proposed move. Initial private evidence is not attenuated. REASSURE removes the recipient’s switching cost for its next choice only.')
enp('Each participant maintains a joint posterior Jᵢ(x, sⱼ, θⱼ) over the fact, the partner’s finite private-record count and an eight-point parameter dictionary. Actual agents are level-1: they infer a partner represented as a cost-only level-0 chooser. Known public evidence and sent messages condition that partner model; repeated assertions are not treated as independent new private observations. Both actual agents reason at level-1, so their partner models are deliberately approximate rather than recursively correct.')
enp('For a candidate move y, the planning window contains its immediate verification outcome E and the partner’s subsequent response Yⱼ, before any verification following that response. We specify P(E,Yⱼ|x,sⱼ,θⱼ,y) and calculate IGᵢ(y) = I[(x,sⱼ,θⱼ); (E,Yⱼ)|y,Jᵢ]. Choices follow softmax[−(Rᵢ(y)−β IGᵢ(y))/τ], with explicit expected loss R, fixed τ = 0.2 and β = 1 in the main condition. This implements an extrinsic-minus-epistemic objective (Friston et al., 2015), also expressible as expected utility plus information value; it does not identify a uniquely active-inference explanation. The factual likelihood is normalized through ρ_ω = sigmoid[ω logit(ρ)], without altering signal arrival.')
enp('The observation model is zₜ ~ C[fₜ,:]. Restriction and invitation are provisionally anchored to LOCK and SEEK; C has independent Dirichlet row posteriors given 15 and 20 archived primary-label cases plus 0.5 per cell. A single matrix is held fixed within each posterior-predictive dialogue and integrated by the observer. There is no direct dependence of labels on θ once form is given. No natural-language text was generated or re-annotated. This bridge operationalizes two forms, not the entire regime ontology, and historical primary-label counts do not establish construct validity or transportability to the target corpus.')

heading(10,'English insertion: results and boundaries')
enp('Computational evaluation and scope of the claims')
enp('Dynamics used 2,000 shared random worlds per condition, sixteen profile pairings, cost-only and verification-only-information controls, targeted interventions and sensor-reliability sensitivity analyses. With normal verification sensitivity (ω = 1), the access-protective pair reduced mean delivered verifications from 3.646 to 2.159 and mean posterior probability of the true fact from 0.885 to 0.838, relative to the reference pair. The paired truth-probability difference was −0.047 (95% Monte Carlo interval −0.054 to −0.040). In contrast, the high-switching-cost pair retained public disagreement in 78.9% of worlds while achieving truth probability 0.907. Public persistence therefore did not imply failed private belief revision in this model. The access-protective pair still improved from an initial truth probability of 0.679: updating was reduced rather than abolished.')
enp('The access-protection effect survived removal of information value: with β = 0, the reference and access-protective pairs had truth probabilities 0.867 and 0.831, respectively. Information value changed behavior but was not necessary for the core reduction in verification. Forcing inviting forms from the eighth move increased the access-protective pair’s truth probability to 0.891. In a high-cost/reference pair, one effective REASSURE reduced final public disagreement from 44.85% under the local inactive-message control to 18.75%, without improving factual learning. These intervention semantics are stipulated features of the simulator, not treatment effects in people.')
enp(f'The external-observer pilot used 96 independent dialogues, balanced across four generating A profiles, with B generated from the reference profile but treated as unknown by the observer. All channels included coarse acts and past verification; additional channels exposed noisy labels or true forms. Labels reduced mean pre-outcome verification-arrival log loss from {ov.loc["actions_evidence","arrival_logloss"]:.3f} to {ov.loc["labels_evidence","arrival_logloss"]:.3f}, a paired improvement of {arr["mean"]:.4f} nats (95% Monte Carlo interval {arr.lo:.4f} to {arr.hi:.4f}). The next-act log-loss gain was {act["mean"]:+.4f} nats ({act.lo:+.4f} to {act.hi:+.4f}). A-profile accuracy was {pct(ov.loc["actions_evidence","correct_A"])} from acts plus evidence and {pct(ov.loc["labels_evidence","correct_A"])} with labels. Improved prediction of verification must therefore be distinguished from mechanism recovery. These scores concern a new task and cannot be inserted as an additional row in the earlier recovery table without revising its design and caption.')
enp('Independent scalar and vector calculations agreed within 2.7×10⁻¹⁵ for choice probabilities. Bayesian averaging, role symmetry, bounded information from finite private records, current-position causality, local-sham timing, Dirichlet integration and null-channel checks passed. The known-form observer is exact on its finite parameter/private-count grid. Hidden forms require a stratified particle approximation: the main run used eight paths in each of 144 static strata, with K = 4 and 32 sensitivity on sixteen dialogues and short exact path enumeration. On those sixteen worlds, the label gain in arrival prediction was 0.0771 nats at K = 8 and 0.0832 at K = 32; individual posterior components changed by up to 0.124. Parameter classifications remain preliminary, and these checks do not validate psychological interpretation.')
enp('The model studies factual belief revision and inference over a fixed partner dictionary. It does not learn its likelihood structure or model long-term generative-model acquisition. The form-to-verification relation, ontology anchoring, private-record reliability and reassurance effect require empirical calibration. A human study should measure beliefs independently of public statements, manipulate information quality and public commitment costs separately, and evaluate predictions on held-out dyads using prefix-only annotations. Physiological maintenance costs and allostatic load remain untested: neither c nor γ is an allostatic measure. The present contribution is a formal, testable bridge between annotation and a specified interaction process, with explicit limits on identification.')

heading(11,'Источники и воспроизведение')
add('Первичные статьи, на которые опирается вводная связка','h2')
refs=[
('1','Friston, K., &amp; Frith, C. (2015a). A Duet for one. Consciousness and Cognition, 36, 390–405.','https://doi.org/10.1016/j.concog.2014.12.003'),
('2','Friston, K. J., &amp; Frith, C. D. (2015b). Active inference, communication and hermeneutics. Cortex, 68, 129–143.','https://doi.org/10.1016/j.cortex.2015.03.025'),
('3','Friston, K. J., Parr, T., Yufik, Y., Sajid, N., Price, C. J., &amp; Holmes, E. (2020). Generative models, linguistic communication and active inference. Neuroscience &amp; Biobehavioral Reviews, 118, 42–64.','https://doi.org/10.1016/j.neubiorev.2020.07.005'),
('4','Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., &amp; Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6(4).','https://doi.org/10.1080/17588928.2015.1020053')]
for n,title,url in refs:add(f'[{n}] {title}<br/><link href="{url}" color="{TEAL}">{url}</link>','small')
add('Что находится в воспроизводимом пакете','h2')
table([['Файл / каталог','Назначение'],['MODEL_SPEC.md и protocol.json','Полная спецификация, параметры, порядок ходов и начальные seeds.'],['model.py / observer.py / channel.py','Диада; вывод внешнего наблюдателя; канал разметки.'],['validate_independent.py','Скалярные вычисления, перебор коротких путей и причинные контроли.'],['run_dynamics.py / run_observer.py','Запуск симуляций и всех основных сравнений.'],['results/','Полные таблицы, отдельные миры наблюдателя, интервалы, проверки численной устойчивости.'],['report/Proposed_hybrid_sections_EN.txt','Английская вставка со страниц 9–10 для последующего редактирования статьи.']],[222,277])
add('Режим воспроизведения','h2')
add('Python, NumPy, SciPy и pandas; без API и внешних языковых моделей. Команды и версии указаны в README.md и environment.json. Наблюдатель со скрытыми формами — наиболее длительная стадия. Архив включает уже рассчитанные результаты.')
add('Обозначения в CSV: reference — базовый; attenuated — ω = 0,3; concession_cost — c = 1,4; access_protection — γ = 2. В остальных координатах параметры базовые (1, 0, 0). All — все миры; both_own — исходно несогласное подмножество.','small')
add('Авторская формулировка новизны должна оставаться узкой: «Мы задаём и проверяем в симуляции явный канал между формой взаимодействия, доступом к проверке и наблюдаемой разметкой». Приоритет в отношении всей области защитной речи здесь не установлен.','small')

plain='\n\n'.join(en)+'\n\nReferences\n'+ '\n'.join(f'[{n}] {html.unescape(title)} {url}' for n,title,url in refs)+'\n'
(OUT/'Proposed_hybrid_sections_EN.txt').write_text(plain)
pdf=OUT/'Mindloom_hybrid_v08_RU_EN_2026-09-30.pdf'
SimpleDocTemplate(str(pdf),pagesize=(595.276,841.89),rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=57,title='Mindloom Hybrid v0.8: reciprocal verification dyad',author='Mindloom — research working report').build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
