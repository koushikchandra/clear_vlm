#!/usr/bin/env python
"""
Bootstrap 95% confidence intervals for ECE (and ECE deltas vs intact) on a
run_eval .jsonl, optionally within a question-type stratum (closed/open/all).

Small medical splits like VQA-RAD (n=451; closed n~251, open n~200) make a
single ECE point estimate noisy, so we resample items with replacement and
report the 2.5/97.5 percentile interval. This backs the RQ3 claim that
closed-question calibration collapses under evidence removal.
"""
import argparse, json
import numpy as np
from score import match, norm, ece


def qtype(gold):
    return "closed" if norm(gold) in {"yes", "no"} else "open"


def ece_of(correct, confs):
    e, _ = ece(list(confs), list(correct))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--stratum", default="all", choices=["all", "closed", "open"])
    ap.add_argument("--nboot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pred)]
    if args.stratum != "all":
        rows = [r for r in rows if qtype(r["gold"]) == args.stratum]
    conds = [c for c in ("intact", "blank", "mismatch") if f"{c}_answer" in rows[0]]

    # per-item correctness + confidence arrays per condition
    C = {c: np.array([match(r[f"{c}_answer"], r["gold"]) for r in rows], float)
         for c in conds}
    P = {c: np.array([r[f"{c}_conf"] for r in rows], float) for c in conds}
    n = len(rows)

    rng = np.random.default_rng(args.seed)
    boot_idx = [rng.integers(0, n, n) for _ in range(args.nboot)]

    def ci(vals):
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    print(f"\n== {args.pred}  stratum={args.stratum}  n={n}  "
          f"nboot={args.nboot} ==")
    point = {}
    for c in conds:
        pt = ece_of(C[c], P[c]); point[c] = pt
        bs = [ece_of(C[c][ix], P[c][ix]) for ix in boot_idx]
        lo, hi = ci(bs)
        print(f"  ECE[{c:<8}] = {pt:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

    for c in ("blank", "mismatch"):
        if c in conds and "intact" in conds:
            dpt = point[c] - point["intact"]
            bs = [ece_of(C[c][ix], P[c][ix]) - ece_of(C["intact"][ix], P["intact"][ix])
                  for ix in boot_idx]
            lo, hi = ci(bs)
            sig = "yes" if lo > 0 else "no"
            print(f"  dECE[{c} - intact] = {dpt:+.3f}   95% CI [{lo:+.3f}, "
                  f"{hi:+.3f}]   >0: {sig}")


if __name__ == "__main__":
    main()
