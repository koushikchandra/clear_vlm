#!/usr/bin/env python
"""
Reliability diagrams from score.py metrics JSON(s).

Each metrics file contributes one panel; within a panel we draw one curve per
image condition (intact/blank/mismatch): accuracy vs. confidence per bin.
Perfect calibration is the diagonal; curves BELOW the diagonal = overconfidence.
"""
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_STYLE = {
    "intact":   ("#1b9e77", "o", "Intact"),
    "blank":    ("#d95f02", "s", "Blank"),
    "mismatch": ("#7570b3", "^", "Mismatch"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True,
                    help="one or more *_metrics.json")
    ap.add_argument("--titles", nargs="+", default=None,
                    help="panel titles (match --metrics order)")
    ap.add_argument("--ncols", type=int, default=0,
                    help="panels per row (0 = single row of all panels)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = len(args.metrics)
    ncols = args.ncols if args.ncols > 0 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:            # hide any unused panels
        ax.axis("off")
    titles = args.titles or [f"panel{i}" for i in range(n)]

    for ax, mfile, title in zip(flat, args.metrics, titles):
        rep = json.load(open(mfile))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)  # ideal
        for cond, (color, mk, lab) in COND_STYLE.items():
            if cond not in rep["conditions"]:
                continue
            bins = rep["conditions"][cond]["reliability_bins"]
            xs = [b[2] for b in bins if b[1] is not None]  # mean conf
            ys = [b[1] for b in bins if b[1] is not None]  # accuracy
            ece = rep["conditions"][cond]["ece"]
            ax.plot(xs, ys, marker=mk, color=color, ms=4, lw=1.4,
                    label=f"{lab} (ECE={ece:.2f})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
