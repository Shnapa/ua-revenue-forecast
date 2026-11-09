import json
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm


root = Path(__file__).resolve().parent
nb = json.loads((root / "ua_revenue_forecast.ipynb").read_text())
ns = {"np": np, "pl": pl, "sm": sm}

# One implementation of naive models, one implementation of M1, and the shared
# metric/model feature definitions used by M2/M3.
for index in [43, 45, 60]:
    exec(compile("".join(nb["cells"][index]["source"]), f"cell-{index}", "exec"), ns)

reports = pl.read_parquet(root / "out/reports_all_v2.parquet")
panel = pl.read_parquet(root / "out/panel_v2.parquet")
flags = ns["flags_for_panel"](ns["add_anomaly_flags"](reports))
panel = ns["attach_flags"](panel, flags)
panel.write_parquet(root / "out/panel_v2_flagged.parquet")

results_m2, m2_predictions = ns["evaluate_m2"](panel, alpha=1.0)
results_m3, m3_predictions, m3_iterations = ns["evaluate_m3"](panel)

rows = []
prediction_frames = []
models = ["N1", "N2", "N3", "M1", "M2", "M3"]
for fold, (train_years, test_year) in ns["FOLDS"].items():
    train = panel.filter(pl.col("target_year").is_in(train_years))
    pred = ns["predict_naive"](m2_predictions[fold])
    m1 = ns["fit_m1"](train)
    pred = pred.with_columns(pl.Series("M1", ns["predict_m1"](m1, pred)))
    pred = pred.join(
        m3_predictions[fold].select("edrpou", "t", "M3"),
        on=["edrpou", "t"], how="left", validate="1:1",
    ).with_columns(pl.lit(fold).alias("fold"))
    prediction_frames.append(pred)
    for population, subset in (
        ("all", pred),
        ("clean", pred.filter(~pl.col("has_anomaly"))),
    ):
        for model in models:
            rows.append({
                "model": model, "fold": fold, "population": population,
                **ns["regression_metrics"](subset["z_next"], subset[model], subset["N1"]),
            })

comparison = pl.DataFrame(rows)
predictions = pl.concat(prediction_frames, how="vertical_relaxed")
comparison.write_parquet(root / "out/comparison_v2.parquet")
predictions.write_parquet(root / "out/predictions_v2.parquet")

summary = comparison.group_by("population", "model").agg(
    pl.col("MAE_log").mean().alias("mean_MAE_log"),
    pl.col("Skill").min().alias("min_Skill"),
).sort("population", "mean_MAE_log")

print("M3 iterations", m3_iterations)
print(comparison.select("population", "model", "fold", "MAE_log", "Skill", "Bias", "n")
      .sort("population", "fold", "model"))
print(summary)
