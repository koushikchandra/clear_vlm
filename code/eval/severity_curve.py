#!/usr/bin/env python
"""
Corruption-severity analysis from a severity-sweep .jsonl (conditions like
intact, blur@1, blur@3, noise@0.05, ...). Computes accuracy / mean-confidence /
ECE per severity and plots them vs. severity for blur and noise families.
A faithful model's confidence should fall as corruption increases.
"""
import argparse, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from score import match, ece


def cond_metrics(rows, cond):
    correct = [match(r[f"{cond}_answer"], r["gold"]) for r in rows]
    confs = [r[f"{cond}_conf"] for r in rows]
    e, _ = ece(confs, correct)
    return float(np.mean(correct)), float(np.mean(confs)), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ncols", type=int, default=None)
    args = ap.parse_args()

    n = len(args.pred)
    ncols = args.ncols or n
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.9 * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    titles = args.titles or args.pred
    csv_rows = []
    # colors per corruption family, consistent across panels
    fam_color = {"blur": "tab:blue", "noise": "tab:orange"}

    for ax, pf, title in zip(flat, args.pred, titles):
        rows = [json.loads(l) for l in open(pf)]
        conds = [c[:-7] for c in rows[0] if c.endswith("_answer")]
        fam = defaultdict(list)  # family -> [(severity, cond)]
        base = None
        for c in conds:
            if c == "intact":
                base = c
            elif "@" in c:
                f, s = c.split("@"); fam[f].append((float(s), c))
        for f in fam:
            fam[f].sort()

        # intact reference
        acc0, conf0, ece0 = cond_metrics(rows, base) if base else (None, None, None)
        for f, items in fam.items():
            # ORDINAL x-axis: 0 = intact, 1,2,3 = the three severity levels.
            # Blur (radius 1/3/5) and noise (sigma .05/.1/.2) share positions so
            # neither family is crammed near zero (see caption for real values).
            xs = list(range(len(items) + 1)) if base else list(range(1, len(items) + 1))
            accs, confs = ([acc0], [conf0]) if base else ([], [])
            for s, c in items:
                a, cf, e = cond_metrics(rows, c)
                accs.append(a); confs.append(cf)
                csv_rows.append((title, f, s, a, cf, e))
            col = fam_color.get(f)
            ax.plot(xs, confs, marker="o", color=col, label=f"{f}: conf.")
            ax.plot(xs, accs, marker="s", ls="--", color=col, label=f"{f}: acc.")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Corruption level"); ax.set_ylim(0, 1)
        ax.set_xticks(range(4))
        ax.grid(alpha=0.25); ax.legend(fontsize=6, ncol=2, loc="lower left")
    for ax in flat[n:]:
        ax.axis("off")
    for r in range(nrows):
        axes[r][0].set_ylabel("Value")
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"[wrote] {args.out}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["panel", "family", "severity", "acc", "mconf", "ece"])
            w.writerows(csv_rows)
        print(f"[wrote] {args.csv}")


if __name__ == "__main__":
    main()
