#!/usr/bin/env python
"""
Compute paper metrics from run_eval.py output (.jsonl).

Metrics per image condition (intact / blank / mismatch):
  - accuracy            : normalized answer match vs gold
  - ECE                 : expected calibration error (confidence vs correctness)
  - hi_conf_err         : fraction of conf>tau answers that are wrong (tau=0.8)
Cross-condition:
  - image_drop_consistency(blank)    : P(blank_ans == intact_ans)
  - image_drop_consistency(mismatch) : P(mismatch_ans == intact_ans)
    (high => model ignores the image; the hallucination signal)
Also dumps reliability-curve bins for plotting.
"""
import argparse, json
from collections import defaultdict
import numpy as np


def norm(s):
    return " ".join(str(s).lower().strip().replace("?", "").split())


def match(pred, gold):
    p, g = norm(pred), norm(gold)
    if p == g:
        return True
    # closed-set yes/no leniency
    if g in {"yes", "no"}:
        return p.startswith(g)
    # open: gold appears as a phrase in prediction
    return g in p and len(g) > 2


def ece(confs, correct, n_bins=10):
    confs, correct = np.asarray(confs), np.asarray(correct, float)
    if len(confs) == 0:
        return float("nan"), []
    edges = np.linspace(0, 1, n_bins + 1)
    e, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi) if hi < 1 else (confs > lo) & (confs <= hi + 1e-9)
        if m.sum() == 0:
            bins.append((float((lo + hi) / 2), None, None, 0)); continue
        acc, conf = correct[m].mean(), confs[m].mean()
        e += (m.sum() / len(confs)) * abs(acc - conf)
        bins.append((float((lo + hi) / 2), float(acc), float(conf), int(m.sum())))
    return float(e), bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="run_eval .jsonl")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", default=None, help="optional metrics .json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred)]
    conds = [c for c in ("intact", "blank", "mismatch")
             if f"{c}_answer" in rows[0]]

    report = {"n": len(rows), "conditions": {}}
    for c in conds:
        correct = [match(r[f"{c}_answer"], r["gold"]) for r in rows]
        confs = [r[f"{c}_conf"] for r in rows]
        e, bins = ece(confs, correct)
        hi = [(cf > args.tau) for cf in confs]
        hi_err = (sum(1 for k, ok in zip(hi, correct) if k and not ok)
                  / max(1, sum(hi)))
        report["conditions"][c] = {
            "accuracy": float(np.mean(correct)),
            "mean_conf": float(np.mean(confs)),
            "ece": e,
            "hi_conf_err_rate": float(hi_err),
            "reliability_bins": bins,
        }

    # image-drop consistency (needs intact as reference)
    if "intact" in conds:
        for c in ("blank", "mismatch"):
            if c in conds:
                same = [norm(r[f"{c}_answer"]) == norm(r["intact_answer"])
                        for r in rows]
                report["conditions"][c]["consistency_vs_intact"] = float(np.mean(same))

    # console summary
    print(f"\n== {args.pred}  (n={report['n']}) ==")
    hdr = f"{'cond':<9}{'acc':>7}{'mconf':>8}{'ECE':>7}{'hiErr':>8}{'consist':>9}"
    print(hdr); print("-" * len(hdr))
    for c in conds:
        m = report["conditions"][c]
        print(f"{c:<9}{m['accuracy']:>7.3f}{m['mean_conf']:>8.3f}"
              f"{m['ece']:>7.3f}{m['hi_conf_err_rate']:>8.3f}"
              f"{m.get('consistency_vs_intact', float('nan')):>9.3f}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
