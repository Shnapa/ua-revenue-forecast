import json
from pathlib import Path


root = Path(__file__).resolve().parent
old = json.loads((root / "ua_revenue_forecast.ipynb").read_text())


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


model_defs = "".join(old["cells"][60]["source"])
model_defs = model_defs.replace("from sklearn.linear_model import LinearRegression, Ridge", "from sklearn.linear_model import Ridge")
start = model_defs.index("def add_m1_prediction")
end = model_defs.index("def segment_metrics")
model_defs = model_defs[:start] + model_defs[end:]

cells = [
md("""# Прогнозування доходу українських підприємств

Єдиний виконуваний ноутбук: завантаження готової панелі, EDA, часовий backtesting, моделі N1–N3 і M1–M3, сегменти та висновки."""),

md("""# 0. Дані та налаштування

Ноутбук не розбирає сирі XML. Він читає одну підготовлену `panel_v1.parquet`, у якій уже об’єднано базові річні зрізи 2021–2024 років із пізніми поданнями, виконано дедуплікацію та додано прапорці аномалій. Цілі обмежені 2024 роком; 2025 рік не використовується."""),

code("""!pip -q install -r requirements.txt
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from huggingface_hub import hf_hub_download

ROOT = Path('/content') if Path('/content').exists() else Path('.')
OUT = ROOT / 'out'
OUT.mkdir(parents=True, exist_ok=True)
HF_REPO = 'Shnnapka/fin-zvit-ua'
PANEL_NAME = 'panel_v1.parquet'

local_panel = OUT / PANEL_NAME
if not local_panel.exists():
    source = hf_hub_download(HF_REPO, f'panel/{PANEL_NAME}', repo_type='dataset', token=False)
    panel = pl.read_parquet(source)
else:
    panel = pl.read_parquet(local_panel)

required = {'edrpou', 't', 'target_year', 'z_t', 'z_lag', 'z_next', 'dz_next',
            'flag_units_x1000', 'flag_balance_mismatch', 'flag_extreme_growth',
            'flag_zero_revenue', 'has_anomaly'}
assert not required.difference(panel.columns), 'panel_v1.parquet has an outdated schema'
assert panel['target_year'].max() == 2024 and panel['target_year'].min() == 2022
f1_lag_missing = panel.filter(pl.col('target_year') == 2022)['z_lag'].is_null().mean()
assert f1_lag_missing < 0.20, 'Outdated panel: F1 still has no 2020 lag; publish cumulative panel_v1'

pl.Config.set_tbl_rows(30)
pl.Config.set_float_precision(4)
print(f'panel: {panel.height:,} rows; companies: {panel["edrpou"].n_unique():,}')
print(panel.group_by('target_year').agg(
    pl.len().alias('rows'),
    pl.col('z_lag').is_null().mean().alias('share_lag_missing'),
).sort('target_year'))"""),

md("""**Висновок.** Панель містить фактичні ознаки за 2021 рік і лаг доходу за 2020 рік із графи «попередній рік» у звіті 2021 року. У F1 лаг відсутній лише для окремих звітів, а не для всього фолду. Перевірка версії зупиняє виконання, якщо завантажено старий parquet."""),

md("""# 1. EDA та прапорці"""),

code("""FLAG_COLUMNS = ['flag_units_x1000', 'flag_balance_mismatch',
                'flag_extreme_growth', 'flag_zero_revenue']

coverage = (panel.group_by('target_year', 'form').len()
            .pivot(on='form', index='target_year', values='len').sort('target_year'))
revenue_summary = panel.group_by('target_year').agg(
    pl.col('y_t').median().alias('median_y_t'),
    pl.col('z_t').median().alias('median_z_t'),
    pl.col('dz_next').median().alias('median_dz_next'),
    (pl.col('z_next') == 0).mean().alias('share_z_next_zero'),
).sort('target_year')
flag_summary = panel.select(
    *[pl.col(c).mean().alias(c) for c in FLAG_COLUMNS],
    pl.col('has_anomaly').mean().alias('has_anomaly'),
)
display(coverage)
display(revenue_summary)
display(flag_summary)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for year in panel['target_year'].unique().sort().to_list():
    values = panel.filter(pl.col('target_year') == year)['z_t'].to_numpy()
    axes[0].hist(values, bins=80, density=True, histtype='step', label=str(year))
axes[0].set(title='Розподіл log1p доходу', xlabel='z_t', ylabel='Щільність')
axes[0].legend()
counts_plot = coverage.to_pandas().set_index('target_year')
counts_plot.plot(kind='bar', stacked=True, ax=axes[1], title='Покриття за формами')
axes[1].set(xlabel='Цільовий рік', ylabel='Рядки')
plt.tight_layout()
plt.show()"""),

md("""**Висновок.** Розподіл доходу має довгий правий хвіст, тому моделі оцінюються на шкалі `log1p`. Дані річні, отже сезонність усередині року не спостерігається і для цієї задачі не визначена. `flag_units_x1000` використовується лише для EDA та post-hoc clean-оцінки, бо потребує повтору показника зі звіту `t+1`; `flag_zero_revenue` після умови `y_t > 0` є константою. Обидва прапорці виключені з модельних ознак."""),

md("""# 2. Валідація, ознаки та моделі

Офіційні фолди: F1 — train 2022, test 2023; F2 — train 2022–2023, test 2024. Усі статистики preprocessing навчаються лише на outer-train. M3 використовує 10% ЄДРПОУ з outer-train для early stopping, а потім перенавчається на всьому outer-train."""),

code(model_defs + """

def sector_growth(df):
    growth = (df.filter(pl.col('z_lag').is_not_null())
              .with_columns((pl.col('z_t') - pl.col('z_lag')).alias('dz_t')))
    by_sector = growth.group_by('t', 'kved_section').agg(pl.col('dz_t').median().alias('g_sec'))
    by_all = growth.group_by('t').agg(pl.col('dz_t').median().alias('g_all'))
    return by_sector, by_all


def predict_naive(test):
    by_sector, by_all = sector_growth(test)
    return (test.join(by_sector, on=['t', 'kved_section'], how='left')
            .join(by_all, on='t', how='left')
            .with_columns(
                pl.col('z_t').alias('N1'),
                (pl.col('z_t') + pl.coalesce('g_sec', 'g_all', pl.lit(0.0))).alias('N2'),
                pl.when(pl.col('z_lag').is_not_null())
                  .then((pl.col('z_t') + pl.col('z_lag')) / 2)
                  .otherwise(pl.col('z_t')).alias('N3'),
            ))


def fit_m1(train):
    return sm.OLS(train['z_next'].to_numpy(), sm.add_constant(train['z_t'].to_numpy())).fit()


def predict_m1(model, test):
    return model.predict(sm.add_constant(test['z_t'].to_numpy()))


MODELS = ['N1', 'N2', 'N3', 'M1', 'M2', 'M3']


def evaluate_six_models(panel):
    rows, frames, iterations = [], [], {}
    for fold, (train_years, test_year) in FOLDS.items():
        train = panel.filter(pl.col('target_year').is_in(train_years))
        test = panel.filter(pl.col('target_year') == test_year)
        assert train['target_year'].max() < test_year

        pred = predict_naive(test)
        pred = pred.with_columns(pl.Series('M1', predict_m1(fit_m1(train), pred)))
        m2 = RidgeDeltaModel(alpha=1.0).fit(train)
        pred = pred.with_columns(pl.Series('M2', m2.predict(test)))
        m3 = LightGBMDeltaModel(seed=42).fit(train)
        pred = pred.with_columns(pl.Series('M3', m3.predict(test)), pl.lit(fold).alias('fold'))
        iterations[fold] = m3.best_iteration_
        frames.append(pred)

        for population, subset in (
            ('all', pred),
            ('clean', pred.filter(~pl.col('has_anomaly'))),
        ):
            for model in MODELS:
                rows.append({'population': population, 'model': model, 'fold': fold,
                             **regression_metrics(subset['z_next'], subset[model], subset['N1'])})
    return pl.DataFrame(rows), pl.concat(frames, how='vertical_relaxed'), iterations"""),

md("""# 3. Єдиний harness: шість моделей, дві популяції"""),

code("""assert not set(NUMERIC_FEATURES) & FUTURE_DERIVED_FEATURES
assert not set(NUMERIC_FEATURES) & EDA_ONLY_FLAGS
assert panel['flag_zero_revenue'].n_unique() == 1 and panel['flag_zero_revenue'][0] == 0

comparison, predictions, m3_iterations = evaluate_six_models(panel)
summary = (comparison.group_by('population', 'model').agg(
    pl.col('MAE_log').mean().alias('mean_MAE_log'),
    pl.col('Skill').min().alias('min_Skill'),
).sort('population', 'mean_MAE_log'))

display(comparison.select('population', 'fold', 'model', 'MAE_log', 'Skill', 'Bias', 'n')
        .sort('population', 'fold', 'model'))
display(summary)
print('M3 best iterations:', m3_iterations)

comparison.write_parquet(OUT / 'comparison_six_models.parquet')
predictions.write_parquet(OUT / 'predictions_six_models.parquet')"""),

md("""**Висновок.** На актуальній панелі N1 має найменший середній `MAE_log` на всій і чистій популяціях. Жодна навчена модель не має `Skill > 0` в обох часових фолдах. Отже, фінальним вибором залишається N1; це рішення тепер спирається на ту саму панель, яку читає та перевіряє код."""),

md("""# 4. Діагностики, rolling window та сегменти"""),

code("""# Випадкова GroupKFold — лише діагностика оптимізму, не офіційна оцінка.
cv_source = panel.filter(
    pl.col('target_year').is_in(FOLDS['F2'][0])
    & (pl.col('edrpou').hash(seed=42) % 10 == 0)
)
cv_results = group_cv_m2(cv_source, alphas=(1.0,), n_splits=5)
cv_summary = cv_results.group_by('alpha').agg(
    pl.col('MAE_log').mean().alias('MAE_log_mean'),
    pl.col('MAE_log').std().alias('MAE_log_sd'),
)

# Rolling-window sensitivity for F2: train only on target year 2023.
rolling_train = panel.filter(pl.col('target_year') == 2023)
f2_test = panel.filter(pl.col('target_year') == 2024)
rolling_m2 = RidgeDeltaModel(alpha=1.0).fit(rolling_train)
rolling_m3 = LightGBMDeltaModel(seed=42).fit(rolling_train)
rolling_rows = []
for model, z_hat in {
    'N1': f2_test['z_t'].to_numpy(),
    'M2 rolling': rolling_m2.predict(f2_test),
    'M3 rolling': rolling_m3.predict(f2_test),
}.items():
    rolling_rows.append({'model': model, **regression_metrics(
        f2_test['z_next'], z_hat, f2_test['z_t'])})
rolling_results = pl.DataFrame(rolling_rows)

target_diagnostics = panel.group_by('target_year').agg(
    (pl.col('z_next') == 0).mean().alias('share_z_next_zero'),
    pl.col('z_lag').is_null().mean().alias('share_z_lag_missing'),
    pl.len().alias('n'),
).sort('target_year')

best_model = summary.filter(pl.col('population') == 'all').sort('mean_MAE_log')[0, 'model']
segments = segment_metrics(predictions, best_model)

display(cv_summary)
display(rolling_results)
display(target_diagnostics)
display(segments.sort('segment', 'MAE_log'))

cv_results.write_parquet(OUT / 'cv_m2.parquet')
rolling_results.write_parquet(OUT / 'rolling_window.parquet')
segments.write_parquet(OUT / 'segments.parquet')"""),

md("""**Висновок.** Випадкова GroupKFold змішує роки й використовується лише для демонстрації оптимізму відносно часового backtesting. Rolling-window перевіряє, чи допомагає відмова від шокового 2022 року у train F2. Сегментний аналіз виконується для фактичного переможця офіційної таблиці, а групи з одиничними спостереженнями не інтерпретуються."""),

md("""# 5. Підсумок та обмеження

1. **Дані.** Використано одну версіоновану панель із фактичним 2021 роком, пізніми поданнями та непорожнім лагом 2020 року у F1.
2. **Валідація.** Фінальне рішення приймається лише за F1–F2; GroupKFold є внутрішньою діагностикою.
3. **Моделі.** N1 перемагає N2–N3 і M1–M3 на `all` та `clean`; M2/M3 мають від’ємний Skill щонайменше в одному фолді.
4. **Причина слабкості M2/M3.** Нульові наступні доходи створюють великий лівий хвіст `Δz`, а часовий зсув робить параметри й оптимальну складність нестабільними.
5. **Обмеження.** Є лише два зовнішні фолди й немає окремого тесту на 2022 рік. Наступні кроки — Optuna лише всередині train, ансамбль N1+M3 за правилом позитивного Skill в обох фолдах і узгодження прогнозів із галузевими агрегатами."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(root / "ua_revenue_forecast.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
