# Forecasting next-year revenue of Ukrainian companies

The project forecasts the net revenue (line 2000) of every Ukrainian company one year ahead from its annual financial statements. All work is in [ua_revenue_forecast.ipynb](ua_revenue_forecast.ipynb).

## Notebook

| Section | Content | Owner |
|---|---|---|
| 0. Goal and data | target, population, deflation, loading the panel | A |
| 1. EDA and preprocessing | coverage, distributions, anomaly flags | B |
| 2. Validation | expanding-window folds, metrics, features, harness, leakage checks | C |
| 3. Baselines | N1-N3, M1-M3 through one harness, decision table, M3 deep dive, segments | A, B, C |
| 4. Bridge to optimization | GroupKFold, rolling window, expm1 bias, next steps | B |
| 5. Conclusions | what works, what does not, what next | A |

Raw XML files are not parsed during a normal run. The notebook reads only the published panel.

## Data

- Source: ["Financial statements of enterprises (annual)"](https://data.gov.ua/dataset/6456ef0c-985c-4bb7-bf9d-16831161b416), State Statistics Service of Ukraine.
- `panel_v1.parquet` has 602,041 rows for 267,960 companies, target years 2022-2024.
- Revenue is in real 2021 prices; the model scale is `z = log1p(y)`.
- The 2025 file is the holdout and is never read.
- The panel is published at [Shnnapka/fin-zvit-ua](https://huggingface.co/datasets/Shnnapka/fin-zvit-ua).

## Running locally

Requirements: Python 3.12 and about 4 GB of free RAM. A full run takes about 10-15 minutes on a laptop CPU, most of it LightGBM.

```bash
cd ua-revenue-forecast
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook ua_revenue_forecast.ipynb
```

Then use **Kernel > Restart Kernel and Run All Cells**. On the first run the notebook downloads `panel/panel_v1.parquet` from Hugging Face (no token needed) and caches it in `out/panel_v1.parquet`. All result tables are also written to `out/`.

On macOS LightGBM needs OpenMP: `brew install libomp`.

In Google Colab, upload the notebook together with `requirements.txt` to `/content` and run all cells; missing packages are installed automatically.

## Validation

| Fold | Train (target year) | Test (target year) |
|---|---|---|
| F1 | 2022 | 2023 |
| F2 | 2022-2023 | 2024 |
| Holdout | 2022-2024 | 2025, not opened |

GroupKFold and a rolling window are used only as diagnostics in section 4. The decision is made on the expanding-window folds F1 and F2.

## Results

`MAE_log` on all rows (lower is better). Skill is relative to N1.

| Model | F1 | F2 | Mean | Works (Skill > 0 in F1 and F2) |
|---|---:|---:|---:|---|
| N1 persistence | **1.026** | **0.905** | **0.966** | reference |
| N2 sector drift | 1.154 | 0.916 | 1.035 | no |
| N3 two-year mean | 1.135 | 1.039 | 1.087 | no |
| M1 panel AR(1) | 1.433 | 1.131 | 1.282 | no |
| M2 Ridge | 1.519 | 1.320 | 1.419 | no |
| M3 LightGBM | 1.243 | 1.063 | 1.153 | no |

N1 is also best on clean rows (mean `MAE_log` 0.892), so it is the final model. The trained models carry the 2022 shock into calm years and forecast declines that did not happen. Trained only on target year 2023 (rolling window), M3 reaches `MAE_log` 0.882 and Skill 0.025 in F2; this is a sensitivity check, not the official result.

## License

The source data is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
