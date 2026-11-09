import json
from pathlib import Path


path = Path("ua_revenue_forecast.ipynb")
nb = json.loads(path.read_text())
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if "counts_plot = coverage.to_pandas()" not in source:
        continue
    source = source.replace(
        "counts_plot = coverage.to_pandas().set_index('target_year')\n"
        "counts_plot.plot(kind='bar', stacked=True, ax=axes[1], title='Покриття за формами')\n",
        "years = coverage['target_year'].to_numpy()\n"
        "bottom = np.zeros(len(years))\n"
        "for form in [c for c in coverage.columns if c != 'target_year']:\n"
        "    values = coverage[form].fill_null(0).to_numpy()\n"
        "    axes[1].bar(years.astype(str), values, bottom=bottom, label=form)\n"
        "    bottom += values\n"
        "axes[1].set_title('Покриття за формами')\n"
        "axes[1].legend()\n",
    )
    cell["source"] = source.splitlines(keepends=True)
    break
else:
    raise RuntimeError("EDA plotting cell not found")
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
