"""Figure for the Topic 6 write-up: popularity baseline vs ALS session-vector.

Reads artifacts/eval_results.json (produced by `uv run evaluate`) and writes
eval-metrics.png next to this script. Grouped bars, one axis, direct value
labels (the baseline bars are near-invisible at linear scale — the labels make
them readable without resorting to a misleading log-scaled bar).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "artifacts" / "eval_results.json"
OUT = HERE / "eval-metrics.png"

SERIES_COLORS = {
    "Popularity baseline": "#1baf7a",
    "ALS session-vector": "#2a78d6",
    "ALS + re-ranker": "#eda100",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"


def main() -> None:
    results = json.loads(RESULTS.read_text())
    k = results["k"]
    metrics = [f"Recall@{k}", f"NDCG@{k}", f"MRR@{k}"]
    series = {
        "Popularity baseline": results["popularity"],
        "ALS session-vector": results["als_session_vector"],
        "ALS + re-ranker": results["als_plus_rerank"],
    }

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(metrics))
    n = len(series)
    width = 0.26
    gap = 0.015  # surface gap between adjacent bars
    bars = []
    for s_i, (label, vals) in enumerate(series.items()):
        offset = (s_i - (n - 1) / 2) * (width + gap)
        heights = [vals[m] for m in ("recall", "ndcg", "mrr")]
        bars.append(
            ax.bar(
                [i + offset for i in x], heights, width,
                label=label, color=SERIES_COLORS[label],
            )
        )
    for group in bars:
        for rect in group:
            ax.annotate(
                f"{rect.get_height():.4f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8.5, color=TEXT_PRIMARY,
            )

    ax.set_xticks(list(x), metrics, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Score (higher is better)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_title(
        f"Leave-last-out evaluation, {results['n_eval_cases']:,} sessions "
        f"(LFM-2b slice, session length {results['session_len']})",
        fontsize=10.5, color=TEXT_PRIMARY, pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
