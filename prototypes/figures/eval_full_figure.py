"""Headline figure for the FINAL report: five systems x three metrics with
bootstrap 95% CI error bars, from artifacts/eval_full.json (`uv run evaluate-full`).

Writes eval-metrics-full.png next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "artifacts" / "eval_full.json"
OUT = HERE / "eval-metrics-full.png"

# Colour follows the entity (same hues as the earlier figure for the shared
# systems; green/violet take the next validated categorical slots).
SYSTEMS = [
    ("popularity", "Popularity", "#1baf7a"),
    ("content_only", "Content-only", "#008300"),
    ("item_knn", "Item-kNN", "#4a3aa7"),
    ("als", "ALS", "#2a78d6"),
    ("als_rerank", "ALS + re-ranker", "#eda100"),
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"


def main() -> None:
    results = json.loads(RESULTS.read_text())
    k = results["k"]
    metrics = [("recall", f"Recall@{k}"), ("ndcg", f"NDCG@{k}"), ("mrr", f"MRR@{k}")]

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n = len(SYSTEMS)
    width = 0.15
    gap = 0.012
    x = range(len(metrics))
    for s_i, (key, label, color) in enumerate(SYSTEMS):
        sys = results["systems"][key]
        offset = (s_i - (n - 1) / 2) * (width + gap)
        heights = [sys[m] for m, _ in metrics]
        errs_lo = [sys[m] - sys["ci95"][m][0] for m, _ in metrics]
        errs_hi = [sys["ci95"][m][1] - sys[m] for m, _ in metrics]
        ax.bar(
            [i + offset for i in x], heights, width, label=label, color=color,
            yerr=[errs_lo, errs_hi], capsize=2,
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 0.9},
        )
        for i, h in zip(x, heights):
            ax.annotate(
                f"{h:.3f}", (i + offset, h), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=6.8, color=TEXT_PRIMARY,
            )

    ax.set_xticks(list(x), [lbl for _, lbl in metrics], fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Score (higher is better)", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_title(
        f"70/10/20 chronological split, {results['n_test_cases']:,} test sessions "
        "(bars: mean, whiskers: bootstrap 95% CI)",
        fontsize=10.5, color=TEXT_PRIMARY, pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncols=2)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
