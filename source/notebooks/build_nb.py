"""Joins 01_prepare.py + 02_train.py into one notebook (no jupytext needed) and runs it."""
import re
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# Smart Crop Advisor (PAI101)\nRun top to bottom. Part 1 prepares and analyses data, part 2 trains, evaluates and exports the models."))
for f in ("01_prepare.py", "01b_synthetic.py", "02_train.py"):
    src = (HERE / f).read_text(encoding="utf-8").replace(
        "Path(__file__).resolve().parents[2]", "Path.cwd().resolve().parents[1] if Path.cwd().name == 'notebooks' else Path.cwd()")
    for chunk in re.split(r"(?m)^# %%", src):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n", 1)[1].splitlines())
            nb.cells.append(nbf.v4.new_markdown_cell(body))
        else:
            nb.cells.append(nbf.v4.new_code_cell(chunk.strip("\n")))
NotebookClient(nb, timeout=1200, resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, HERE / "smart_crop_advisor.ipynb")
print("cells:", len(nb.cells))
