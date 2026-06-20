"""Console entry points wired to pyproject [project.scripts].

`train` runs the training script; `demo`/`explore` open the notebooks via
jupyter lab. Scaffold-time stubs — train() raises until block B2 lands.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


def train() -> None:
    """Run the full ALS training pipeline: load -> filter -> CSR -> fit -> save."""
    from nextrack import data, train as trainmod

    print("Loading + filtering listening events...")
    plays = data.load_events()
    print(f"  filtered playcount rows: {len(plays):,}")

    print("Building user x track CSR matrix...")
    matrix, track_id_to_index, index_to_track_id = data.build_csr(plays)
    print(f"  matrix shape: {matrix.shape}, nnz: {matrix.nnz:,}")

    print("Loading track names...")
    track_names = data.load_track_names()
    print(f"  names loaded: {len(track_names):,}")

    print("Loading tags (this reads the 142MB tag file)...")
    tags = data.load_tags(track_names)
    print(f"  tracks with tags: {len(tags):,}")

    print("Fitting implicit ALS...")
    model = trainmod.train_als(matrix)

    print("Saving artifacts...")
    trainmod.save_artifacts(
        model, index_to_track_id, track_id_to_index, track_names, tags
    )
    print("Done. Artifacts written to prototypes/artifacts/")


def demo() -> None:
    """Open the video demo notebook."""
    subprocess.run([sys.executable, "-m", "jupyterlab", str(_NOTEBOOKS / "02-demo.ipynb")], check=False)


def explore() -> None:
    """Open the data-exploration notebook."""
    subprocess.run([sys.executable, "-m", "jupyterlab", str(_NOTEBOOKS / "01-data-explore.ipynb")], check=False)
