import json
from pathlib import Path


path = Path("ua_revenue_forecast.ipynb")
nb = json.loads(path.read_text())
cell = nb["cells"][2]
source = "".join(cell["source"])
source = source.replace(
    "!pip -q install -r requirements.txt\nfrom pathlib import Path\n",
    "import sys\n!{sys.executable} -m pip -q install -r requirements.txt\nfrom pathlib import Path\n",
)
source = source.replace("from huggingface_hub import hf_hub_download\n", "")
source = source.replace(
    "if not local_panel.exists():\n    source = hf_hub_download",
    "if not local_panel.exists():\n    from huggingface_hub import hf_hub_download\n    source = hf_hub_download",
)
cell["source"] = source.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
