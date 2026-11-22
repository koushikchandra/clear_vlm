#!/usr/bin/env python
"""
Stratified analysis from run_eval .jsonl: split metrics by question type
(closed yes/no vs open-ended). No new inference needed.
"""
import argparse, json
import numpy as np
from score import match, norm, ece


def qtype(gold):
    return "closed" if norm(gold) in {"yes", "no"} else "open"


def block(rows, conds):
    out = {}
    for c in conds:
        correct = [match(r[f"{c}_answer"], r["gold"]) for r in rows]
        confs = [r[f"{c}_conf"] for r in rows]
        e, _ = ece(confs, correct)
        row = {"n": len(rows), "acc": float(np.mean(correct)),
               "mconf": float(np.mean(confs)), "ece": e}
        if c != "intact":
            same = [norm(r[f"{c}_answer"]) == norm(r["intact_answer"]) for r in rows]
            row["consist"] = float(np.mean(same))
        out[c] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.pred)]
    conds = [c for c in ("intact", "blank", "mismatch") if f"{c}_answer" in rows[0]]

    strata = {"all": rows,
              "closed": [r for r in rows if qtype(r["gold"]) == "closed"],
              "open":   [r for r in rows if qtype(r["gold"]) == "open"]}
    report = {s: block(rs, conds) for s, rs in strata.items() if rs}

    for s, rep in report.items():
        print(f"\n[{s}]  n={report[s][conds[0]]['n']}")
        print(f"{'cond':<9}{'acc':>7}{'mconf':>8}{'ece':>7}{'consist':>9}")
        for c in conds:
            m = rep[c]
            print(f"{c:<9}{m['acc']:>7.3f}{m['mconf']:>8.3f}{m['ece']:>7.3f}"
                  f"{m.get('consist', float('nan')):>9.3f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
