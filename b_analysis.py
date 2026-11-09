"""Person B deliverables: anomaly flags, Ridge M2, metrics and segment analysis.

The functions are deliberately notebook-friendly and operate on Polars DataFrames.
No function reads 2025 data.  Preprocessing statistics are fitted separately inside
every training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FOLDS = {"F1": ([2022], 2023), "F2": ([2022, 2023], 2024)}
NUMERIC_FEATURES = [
    "z_t",
    "z_lag",
    "dz_t",
    "log_assets",
    "asinh_equity",
    "leverage",
    "asset_turnover",
    "sales_margin",
    "sector_growth",
    "log_employees",
    "flag_units_x1000",
    "flag_balance_mismatch",
    "flag_extreme_growth",
    "flag_zero_revenue",
]
CATEGORICAL_FEATURES = ["form", "kved_section", "obl"]
FLAG_COLUMNS = [
    "flag_units_x1000",
    "flag_balance_mismatch",
    "flag_extreme_growth",
    "flag_zero_revenue",
]
KVED_SECTIONS = [
    ("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35), ("E", 36, 39),
    ("F", 41, 43), ("G", 45, 47), ("H", 49, 53), ("I", 55, 56), ("J", 58, 63),
    ("K", 64, 66), ("L", 68, 68), ("M", 69, 75), ("N", 77, 82), ("O", 84, 84),
    ("P", 85, 85), ("Q", 86, 88), ("R", 90, 93), ("S", 94, 96), ("T", 97, 98),
    ("U", 99, 99),
]


@dataclass(frozen=True)
class AnomalyThresholds:
    balance_relative_tolerance: float = 0.01
    balance_absolute_tolerance: float = 1.0
    scale_ratio_low: float = 900.0
    scale_ratio_high: float = 1100.0
    extreme_growth_factor: float = 10.0


def kved_section_expr(column: str = "kved") -> pl.Expr:
    division = pl.col(column).str.extract(r"^(\d{1,2})", 1).cast(pl.Int32, strict=False)
    result = pl.lit(None, dtype=pl.String)
    for section, low, high in reversed(KVED_SECTIONS):
        result = pl.when(division.is_between(low, high)).then(pl.lit(section)).otherwise(result)
    return result


def _factor_1000(a: str, b: str, thresholds: AnomalyThresholds) -> pl.Expr:
    """Whether non-zero columns a and b differ by approximately a factor of 1000."""
    ratio = pl.col(a).abs() / pl.col(b).abs()
    return (
        pl.col(a).is_not_null()
        & pl.col(b).is_not_null()
        & (pl.col(a) != 0)
        & (pl.col(b) != 0)
        & (
            ratio.is_between(thresholds.scale_ratio_low, thresholds.scale_ratio_high)
            | ratio.is_between(
                1 / thresholds.scale_ratio_high,
                1 / thresholds.scale_ratio_low,
            )
        )
    )


def add_anomaly_flags(
    reports: pl.DataFrame,
    thresholds: AnomalyThresholds = AnomalyThresholds(),
) -> pl.DataFrame:
    """Add reproducible flags to parsed reports before panel population filtering.

    The x1000 check compares a value in report t with the repeated previous-year
    value in report t+1.  Balance mismatch is checked on the same filing.  Extreme
    growth uses only values available in the report for year t.
    """
    required = {
        "edrpou", "year", "revenue", "revenue_prev", "assets", "assets_prev",
        "equity", "equity_prev", "lt_liab", "lt_liab_prev", "cur_liab",
        "cur_liab_prev", "net_result", "net_result_prev",
    }
    missing = required.difference(reports.columns)
    if missing:
        raise ValueError(f"reports is missing columns: {sorted(missing)}")

    repeated_variables = ["revenue", "assets", "equity", "lt_liab", "cur_liab", "net_result"]
    next_previous = reports.select(
        "edrpou",
        (pl.col("year") - 1).alias("year"),
        *[pl.col(f"{name}_prev").alias(f"next_{name}_prev") for name in repeated_variables],
    )
    out = reports.join(next_previous, on=["edrpou", "year"], how="left")

    rhs = pl.sum_horizontal("equity", "lt_liab", "cur_liab")
    balance_gap = (pl.col("assets") - rhs).abs()
    balance_tolerance = pl.max_horizontal(
        pl.lit(thresholds.balance_absolute_tolerance),
        pl.col("assets").abs() * thresholds.balance_relative_tolerance,
    )
    balance_available = pl.all_horizontal(
        pl.col("assets").is_not_null(),
        pl.col("equity").is_not_null(),
        pl.col("lt_liab").is_not_null(),
        pl.col("cur_liab").is_not_null(),
    )
    scale_checks = [
        _factor_1000(name, f"next_{name}_prev", thresholds)
        for name in repeated_variables
    ]
    growth_limit = np.log(thresholds.extreme_growth_factor)

    out = out.with_columns(
        pl.any_horizontal(scale_checks).fill_null(False).alias("flag_units_x1000"),
        (balance_available & (balance_gap > balance_tolerance))
        .fill_null(False)
        .alias("flag_balance_mismatch"),
        (
            (pl.col("revenue") > 0)
            & (pl.col("revenue_prev") > 0)
            & ((pl.col("revenue").log() - pl.col("revenue_prev").log()).abs() > growth_limit)
        )
        .fill_null(False)
        .alias("flag_extreme_growth"),
        (pl.col("revenue").fill_null(0) == 0).alias("flag_zero_revenue"),
    ).with_columns(pl.any_horizontal(FLAG_COLUMNS).alias("has_anomaly"))

    return out.drop([f"next_{name}_prev" for name in repeated_variables])


def flags_for_panel(flagged_reports: pl.DataFrame) -> pl.DataFrame:
    """Return the separate (edrpou, t) join requested by the specification."""
    return flagged_reports.select(
        "edrpou",
        pl.col("year").alias("t"),
        *FLAG_COLUMNS,
        "has_anomaly",
    ).unique(["edrpou", "t"], keep="last")


def attach_flags(panel: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
    out = panel.join(flags, on=["edrpou", "t"], how="left")
    return out.with_columns(
        *[pl.col(c).fill_null(False).cast(pl.Int8) for c in FLAG_COLUMNS],
        pl.col("has_anomaly").fill_null(False),
    )


def add_model_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create M2 features using information available by year t only."""
    sector = (
        df.filter(pl.col("z_lag").is_not_null())
        .with_columns((pl.col("z_t") - pl.col("z_lag")).alias("_dz_t"))
        .group_by("t", "kved_section")
        .agg(pl.col("_dz_t").median().alias("sector_growth"))
    )
    country = (
        df.filter(pl.col("z_lag").is_not_null())
        .with_columns((pl.col("z_t") - pl.col("z_lag")).alias("_dz_t"))
        .group_by("t")
        .agg(pl.col("_dz_t").median().alias("country_growth"))
    )
    out = df.join(sector, on=["t", "kved_section"], how="left").join(country, on="t", how="left")
    out = out.with_columns(
        (pl.col("z_t") - pl.col("z_lag")).alias("dz_t"),
        pl.when(pl.col("assets") >= 0).then(pl.col("assets").log1p()).alias("log_assets"),
        pl.col("equity").arcsinh().alias("asinh_equity"),
        pl.when(pl.col("assets") > 0)
        .then(((pl.col("lt_liab") + pl.col("cur_liab")) / pl.col("assets")).clip(0, 5))
        .alias("leverage"),
        pl.when(pl.col("assets") > 0).then(pl.col("y_t") / pl.col("assets")).alias("asset_turnover"),
        pl.when(pl.col("y_t") > 0)
        .then((pl.col("net_result") / pl.col("y_t")).clip(-1, 1))
        .alias("sales_margin"),
        pl.coalesce("sector_growth", "country_growth").alias("sector_growth"),
        pl.when(pl.col("employees") >= 0).then(pl.col("employees").log1p()).alias("log_employees"),
    ).drop("country_growth")
    return out


def regression_metrics(z: Iterable[float], z_hat: Iterable[float], z_n1=None) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    z_hat = np.asarray(z_hat, dtype=float)
    mae = float(np.mean(np.abs(z - z_hat)))
    y = np.expm1(np.clip(z, -50, 50))
    y_hat = np.expm1(np.clip(z_hat, -50, 50))
    positive = y > 0
    result = {
        "MAE_log": mae,
        "Bias": float(np.mean(z_hat - z)),
        "MdAPE": float(np.median(np.abs(y[positive] - y_hat[positive]) / y[positive])) if positive.any() else np.nan,
        "WAPE": float(np.sum(np.abs(y - y_hat)) / np.sum(y)) if np.sum(y) else np.nan,
        "n": int(len(z)),
    }
    if z_n1 is not None:
        n1_mae = np.mean(np.abs(z - np.asarray(z_n1, dtype=float)))
        result["Skill"] = float(1 - mae / n1_mae) if n1_mae else np.nan
    return result


class RidgeDeltaModel:
    """M2: Ridge on delta z, with train-only preprocessing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.turnover_cap_: float | None = None
        self.pipeline_: Pipeline | None = None

    @staticmethod
    def _matrix(df: pl.DataFrame) -> np.ndarray:
        return df.select(NUMERIC_FEATURES + CATEGORICAL_FEATURES).to_numpy()

    def _cap_turnover(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.turnover_cap_ is None:
            raise RuntimeError("turnover cap is not fitted")
        return df.with_columns(pl.col("asset_turnover").clip(upper_bound=self.turnover_cap_))

    def fit(self, train: pl.DataFrame) -> "RidgeDeltaModel":
        train = add_model_features(train)
        finite_turnover = train.filter(pl.col("asset_turnover").is_finite())["asset_turnover"]
        self.turnover_cap_ = float(finite_turnover.quantile(0.99)) if len(finite_turnover) else 0.0
        train = self._cap_turnover(train)
        n_num = len(NUMERIC_FEATURES)
        numeric = Pipeline([
            # F1 has no 2020 lag for any row. Keep all-empty columns as zero after
            # imputation so train and test matrices retain the same explicit schema.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric, list(range(n_num))),
            ("categorical", categorical, list(range(n_num, n_num + len(CATEGORICAL_FEATURES)))),
        ])
        self.pipeline_ = Pipeline([
            ("preprocess", preprocess),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.pipeline_.fit(self._matrix(train), train["dz_next"].to_numpy())
        return self

    def predict(self, test: pl.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("model is not fitted")
        test = self._cap_turnover(add_model_features(test))
        delta_hat = self.pipeline_.predict(self._matrix(test))
        return test["z_t"].to_numpy() + delta_hat


def group_cv_m2(
    train: pl.DataFrame,
    alphas: Iterable[float] = (0.1, 1.0, 10.0, 100.0),
    n_splits: int = 5,
) -> pl.DataFrame:
    """Leakage-safe diagnostic CV grouped by EDRPOU, entirely inside outer train."""
    groups = train["edrpou"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two companies")
    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(train.height)
    for alpha in alphas:
        for cv_fold, (fit_idx, val_idx) in enumerate(splitter.split(dummy, groups=groups), 1):
            fit = train[fit_idx]
            val = train[val_idx]
            model = RidgeDeltaModel(alpha=alpha).fit(fit)
            z_hat = model.predict(val)
            rows.append({
                "alpha": float(alpha),
                "cv_fold": cv_fold,
                **regression_metrics(val["z_next"], z_hat, val["z_t"]),
            })
    return pl.DataFrame(rows)


def evaluate_m2(panel: pl.DataFrame, alpha: float = 1.0) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Official expanding-window F1/F2 evaluation on all and clean rows."""
    assert panel["target_year"].max() <= 2024, "2025 holdout found"
    rows: list[dict] = []
    predictions: dict[str, pl.DataFrame] = {}
    for fold, (train_years, test_year) in FOLDS.items():
        train = panel.filter(pl.col("target_year").is_in(train_years))
        test = panel.filter(pl.col("target_year") == test_year)
        assert train["target_year"].max() < test_year
        model = RidgeDeltaModel(alpha=alpha).fit(train)
        pred = test.with_columns(pl.Series("M2", model.predict(test)))
        for population, subset in (
            ("all", pred),
            ("clean", pred.filter(~pl.col("has_anomaly"))),
        ):
            rows.append({
                "model": "M2",
                "fold": fold,
                "population": population,
                **regression_metrics(subset["z_next"], subset["M2"], subset["z_t"]),
            })
        predictions[fold] = pred
    return pl.DataFrame(rows), predictions


def add_m1_prediction(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    model = LinearRegression().fit(train.select("z_t").to_numpy(), train["z_next"].to_numpy())
    return test.with_columns(pl.Series("M1", model.predict(test.select("z_t").to_numpy())))


def segment_metrics(
    predictions: pl.DataFrame,
    prediction_column: str,
    segments: Iterable[str] = ("form", "kved_section"),
) -> pl.DataFrame:
    """MAE_log and Bias by requested segments; small groups stay visible with n."""
    frames = []
    for segment in segments:
        frames.append(
            predictions.group_by(segment).agg(
                (pl.col(prediction_column) - pl.col("z_next")).abs().mean().alias("MAE_log"),
                (pl.col(prediction_column) - pl.col("z_next")).mean().alias("Bias"),
                pl.len().alias("n"),
            ).rename({segment: "segment_value"}).with_columns(
                pl.lit(segment).alias("segment"),
                pl.lit(prediction_column).alias("model"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed").select(
        "model", "segment", "segment_value", "MAE_log", "Bias", "n"
    )
