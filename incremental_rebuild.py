import json
from pathlib import Path

import polars as pl


root = Path(__file__).resolve().parent
nb = json.loads((root / "ua_revenue_forecast.ipynb").read_text())
ns = {}

# Load the notebook's parser and builder definitions, without executing pip.
setup = "".join(nb["cells"][4]["source"])
setup = "\n".join(setup.splitlines()[1:])
exec(compile(setup, "cell-4", "exec"), ns)
for index in [6, 8, 9, 11]:
    exec(compile("".join(nb["cells"][index]["source"]), f"cell-{index}", "exec"), ns)

# The incremental runner keeps yearly frames in memory and writes only final v2
# artifacts, which avoids duplicating ~40 MB per year on a nearly full disk.
builder = "".join(nb["cells"][11]["source"])
builder = builder.replace("    allf.write_parquet(f'{OUT}/reports_{year}.parquet')\n", "")
exec(compile(builder, "cell-11-incremental", "exec"), ns)

# macOS runner: use curl because wget is available in Colab but not installed here.
def download_with_curl(name, rid, rev):
    dst = f'{ns["RAW"]}/{name}.zip'
    if ns["os"].path.exists(dst + '.ok'):
        return dst
    if rev is None:
        url = ns["requests"].get(ns["API"], params={'id': rid}, timeout=60).json()['result']['url']
    else:
        url = f'{ns["BASE"]}/{rid}/revision/{rev}/download'
    print(f'   ↓ {name} ...', flush=True)
    ns["subprocess"].run(['curl', '-fL', '--retry', '5', '-C', '-', '-o', dst, url], check=True)
    ns["zipfile"].ZipFile(dst).close()
    Path(dst + '.ok').touch()
    print(f'   ✓ {name}: {ns["os"].path.getsize(dst) / 1e6:.0f} MB', flush=True)
    return dst


ns["download"] = download_with_curl

all_sources = ns["SOURCES"]
ns["SOURCES"] = {
    2021: all_sources[2021],
    2022: [all_sources[2022][-1]],
    2023: [all_sources[2023][-1]],
    2024: [all_sources[2024][-1]],
}

legacy = pl.read_parquet(root / "out/reports_all.parquet")
assert legacy["year"].unique().sort().to_list() == [2022, 2023, 2024]

# These ignored local artifacts are reproducible from `legacy` and are replaced
# by verified v2 files below. Removing them provides enough room to finish the
# resumable 2021 download; source code and tracked files are untouched.
for name in [
    "reports_all.parquet", "panel_v1.parquet", "anomaly_flags_B.parquet",
    "cv_B.parquet", "results_B.parquet", "results_M3.parquet",
    "comparison_six_models.parquet", "segments_B.parquet",
    "zero_tail_sensitivity_B.parquet", "zero_target_diagnostics_B.parquet",
]:
    candidate = root / "out" / name
    if candidate.exists():
        candidate.unlink()

updates = {}
for year in [2021, 2022, 2023, 2024]:
    updates[year] = ns["build_year"](year)
    for source_name, _, _ in ns["SOURCES"][year]:
        archive = root / "raw" / f"{source_name}.zip"
        marker = Path(str(archive) + ".ok")
        if archive.exists():
            archive.unlink()
        if marker.exists():
            marker.unlink()

reports = ns["latest"](
    pl.concat([legacy, *updates.values()], how="diagonal_relaxed")
)
assert reports["year"].unique().sort().to_list() == [2021, 2022, 2023, 2024]
assert reports.select(pl.struct("edrpou", "year").n_unique()).item() == reports.height
reports.write_parquet(root / "out/reports_all_v2.parquet")

# Build the panel with the exact notebook code, but inject the verified v2 reports.
for index in [22, 24]:
    exec(compile("".join(nb["cells"][index]["source"]), f"cell-{index}", "exec"), ns)
cell26 = "".join(nb["cells"][26]["source"])
cell26 = cell26.replace("rep = load_parquet('reports_all.parquet', 'reports')", "rep = reports")
exec(compile(cell26, "cell-26", "exec"), ns)
exec(compile("".join(nb["cells"][28]["source"]), "cell-28", "exec"), ns)
panel = ns["panel"]
assert panel["target_year"].unique().sort().to_list() == [2022, 2023, 2024]
assert panel.filter(pl.col("target_year") == 2022)["z_lag"].is_not_null().any()
panel.write_parquet(root / "out/panel_v2.parquet")

print("reports_v2", reports.shape, reports.group_by("year").len().sort("year"))
print("panel_v2", panel.shape)
print(panel.group_by("target_year").agg(
    pl.len().alias("n"),
    pl.col("z_lag").is_null().mean().alias("share_lag_missing"),
).sort("target_year"))
