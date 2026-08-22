"""Console entry points wired to pyproject [project.scripts].

`train` runs the full training pipeline (load -> filter -> CSR -> fit -> save);
`demo`/`explore` open the notebooks via JupyterLab.
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
    track_plays = plays.groupby("track_id")["playcount"].sum().to_dict()
    trainmod.save_artifacts(
        model, index_to_track_id, track_id_to_index, track_names, tags,
        track_plays=track_plays,
    )
    print("Done. Artifacts written to prototypes/artifacts/")


def _launch_notebook(name: str) -> None:
    """Launch JupyterLab on a notebook; exit cleanly on Ctrl+C.

    JupyterLab is a long-running server (it serves until stopped), so Ctrl+C is
    the normal way to quit. Without this, the interrupt surfaces as a noisy
    KeyboardInterrupt traceback; we swallow it and print a tidy message instead.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "jupyterlab", str(_NOTEBOOKS / name)],
            check=False,
        )
    except KeyboardInterrupt:
        print("\nJupyterLab server stopped.")


def demo() -> None:
    """Open the video demo notebook."""
    _launch_notebook("02-demo.ipynb")


def explore() -> None:
    """Open the data-exploration notebook."""
    _launch_notebook("01-data-explore.ipynb")
