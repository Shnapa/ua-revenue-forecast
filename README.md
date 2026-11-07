# ua-revenue-forecast

Forecast next-year net revenue (line 2000) of Ukrainian companies from their open annual financial statements.

## Data

- **Source:** State Statistics Service of Ukraine, dataset “Фінансова звітність підприємств (річна)” on [data.gov.ua](https://data.gov.ua/dataset/6456ef0c-985c-4bb7-bf9d-16831161b416), reporting years 2022–2024. 2025 is kept as a holdout and is never read.
- **Processed data:** Hugging Face dataset [Shnnapka/fin-zvit-ua](https://huggingface.co/datasets/Shnnapka/fin-zvit-ua)
  - `reports/`: one row per company and reporting year.
  - `panel/`: the modelling panel (target = log of next-year net revenue in 2021 prices) and a small sample.

Data files are not stored in this repository.

## Repository structure

| File | Contents |
|---|---|
| `01_data.ipynb` | Data download and parsing, panel construction, validation folds, baselines N1–N3 and M1, leakage checks |

More notebooks (features and models) will be added.

## How to run

1. Open `01_data.ipynb` in Google Colab.
2. Keep the default flags `REBUILD = False` and `UPLOAD = False`.
3. Run **Restart & Run All**. It takes about 1–2 minutes and reads the processed data from Hugging Face.

`REBUILD = True` downloads and parses the raw XML again (~30 min, ~4.3 GB). `UPLOAD = True` publishes the files to Hugging Face and asks for a write token.

## Key results

MAE on the log scale (lower is better), test years 2023 and 2024:

| Model | 2023 | 2024 |
|---|---|---|
| N1: persistence | **1.024** | **0.902** |
| N2: sector drift | 1.152 | 0.913 |
| N3: two-year mean | 1.132 | 1.036 |
| M1: panel AR(1), OLS | 1.446 | 1.171 |
| M1-LAD: panel AR(1), median regression | 1.188 | 0.950 |

No model beats persistence. Models trained on 2022 learn the war shock (median real revenue ≈ −37%) and under-predict the recovery. Switching M1 from OLS to LAD cuts its error by about 18%.

## License

Source data: State Statistics Service of Ukraine, published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
