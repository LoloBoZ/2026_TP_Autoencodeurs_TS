"""Retire les sorties calculées des notebooks avant un commit public."""
from pathlib import Path

import nbformat


root = Path(__file__).resolve().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    removed = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            removed += len(cell.outputs)
            cell.outputs = []
            cell.execution_count = None
    if removed:
        nbformat.write(notebook, path)
    print(f"{path.name}: {removed} sorties supprimées")
