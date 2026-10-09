"""Offline analyses on per-item results (steps 5, 6 and 8 of CLAUDE.md).

Usage: python scripts/analyze.py <items.jsonl> <config.yaml> <out.json>

- coverage (step 5): for every crop condition, the share of items whose crop
  window overlaps the evidence region, against the accuracy kept. Evidence =
  oracle box where the dataset has one (rows of the oracle condition store it).
- disagreement (step 6): items where the gaze crop and the object crop barely
  overlap; which one holds the evidence, and which one gets the answer right.
- mistake cue (step 8): on error detection, the distance between the gaze point
  and the object box of the current step, error steps vs correct steps (AUC).
Numbers only; plots are made separately.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import yaml

from eval.stats import bootstrap_ci


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def frac_frames_overlap(boxes_a, boxes_b) -> float:
    return float(np.mean([overlaps(a, b) for a, b in zip(boxes_a, boxes_b)]))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main(items_path: str, cfg_path: str, out_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    rows = [json.loads(l) for l in open(items_path)]
    by = defaultdict(dict)  # item_id -> condition -> row
    for r in rows:
        by[r["item_id"]][r["condition"]] = r
    conds = {c["name"]: c for c in cfg["conditions"]}
    full = next(n for n, c in conds.items() if c["view"] == "full")
    out = {"n_items": len(by)}

    # Evidence region per item (rows carry the raw oracle box where the dataset has one),
    # repeated per frame so it lines up with the per-frame crop boxes.
    evidence = {}
    for i, cs in by.items():
        r = next(iter(cs.values()))
        if "oracle_box" in r:
            evidence[i] = [r["oracle_box"]] * r["n_frames"]

    # --- step 5: coverage vs kept accuracy, per (task category, condition) ---------------------
    cov = []
    if evidence:
        cats = sorted({c for r in rows for c in r["categories"]})
        for cat in cats:
            ids = [i for i in evidence if cat in next(iter(by[i].values()))["categories"] and full in by[i]]
            if not ids:
                continue
            full_acc = np.mean([by[i][full]["correct"] for i in ids])
            for n, c in conds.items():
                if c["view"] not in ("crop", "crop_periphery") or c.get("signal") == "oracle":
                    continue
                have = [i for i in ids if n in by[i]]
                if not have:
                    continue
                # Crop + periphery: rows store only the crop boxes; the periphery always covers everything.
                covered = np.array([frac_frames_overlap(by[i][n]["boxes"], evidence[i]) for i in have])
                corr = np.array([by[i][n]["correct"] for i in have], float)
                cov.append({"category": cat, "condition": n, "signal": c["signal"], "budget": c["budget"],
                            "view": c["view"], "n": len(have), "coverage": float(covered.mean()),
                            "accuracy": float(corr.mean()), "ci95": bootstrap_ci(corr),
                            "kept_vs_full": float(corr.mean() / max(full_acc, 1e-9)),
                            "acc_when_covered": float(corr[covered >= 0.5].mean()) if (covered >= 0.5).any() else None,
                            "acc_when_missed": float(corr[covered < 0.5].mean()) if (covered < 0.5).any() else None})
        # Linear fit of kept accuracy on coverage, crop-only conditions.
        pts = [(p["coverage"], p["kept_vs_full"]) for p in cov if p["view"] == "crop"]
        if len(pts) >= 3:
            x, y = np.array(pts).T
            slope, icpt = np.polyfit(x, y, 1)
            resid = y - (slope * x + icpt)
            r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
            out["coverage_fit"] = {"slope": float(slope), "intercept": float(icpt), "r2": float(r2), "n_points": len(pts)}
    out["coverage"] = cov

    # --- step 6: gaze vs object disagreement ---------------------------------------------------
    dis = []
    for budget in cfg["budgets"]:
        g = next((n for n, c in conds.items() if c["view"] == "crop" and c.get("budget") == budget
                  and c.get("signal") in ("gaze_measured", "gaze_predicted")), None)
        o = next((n for n, c in conds.items() if c["view"] == "crop" and c.get("budget") == budget
                  and c.get("signal") == "object"), None)
        if not (g and o):
            continue
        both = [i for i, cs in by.items() if g in cs and o in cs]
        ious = np.array([np.mean([iou(a, b) for a, b in zip(by[i][g]["boxes"], by[i][o]["boxes"])]) for i in both])
        disagree = [i for i, v in zip(both, ious) if v < 0.1]
        res = {"budget": budget, "gaze_condition": g, "object_condition": o, "n_both": len(both),
               "n_disagree": len(disagree), "share_disagree": len(disagree) / max(len(both), 1),
               "gaze_correct_when_disagree": float(np.mean([by[i][g]["correct"] for i in disagree])) if disagree else None,
               "object_correct_when_disagree": float(np.mean([by[i][o]["correct"] for i in disagree])) if disagree else None}
        if evidence:
            ev = [i for i in disagree if i in evidence]
            res["n_disagree_with_evidence"] = len(ev)
            if ev:
                res["gaze_holds_evidence"] = float(np.mean([frac_frames_overlap(by[i][g]["boxes"], evidence[i]) >= 0.5 for i in ev]))
                res["object_holds_evidence"] = float(np.mean([frac_frames_overlap(by[i][o]["boxes"], evidence[i]) >= 0.5 for i in ev]))
        dis.append(res)
    out["disagreement"] = dis

    # --- step 8: gaze vs step-expected objects as a mistake cue (error detection only) ----------
    mc = []
    for budget in cfg["budgets"]:
        g = next((n for n, c in conds.items() if c["view"] == "crop" and c.get("budget") == budget
                  and c.get("signal") in ("gaze_measured", "gaze_predicted")), None)
        o = next((n for n, c in conds.items() if c["view"] == "crop" and c.get("budget") == budget
                  and c.get("signal") == "object"), None)
        if not (g and o):
            continue
        ids = [i for i, cs in by.items() if g in cs and o in cs and cs[g]["task"] == "error_detection"]
        if not ids:
            continue
        # Gaze point = centre of the gaze crop window; distance to the object window, per frame, mean over frames.
        def dist(i):
            d = []
            for gb, ob in zip(by[i][g]["boxes"], by[i][o]["boxes"]):
                gx, gy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                dx = max(ob[0] - gx, 0, gx - ob[2])
                dy = max(ob[1] - gy, 0, gy - ob[3])
                d.append(np.hypot(dx, dy))
            return float(np.mean(d))
        dd = np.array([dist(i) for i in ids])
        err = np.array([by[i][g]["answer"] == "A" for i in ids])
        mc.append({"budget": budget, "n_steps": len(ids), "n_error_steps": int(err.sum()),
                   "mean_dist_error": float(dd[err].mean()) if err.any() else None,
                   "mean_dist_correct": float(dd[~err].mean()) if (~err).any() else None,
                   "auc_dist_predicts_error": auc(dd[err], dd[~err])})
    out["mistake_cue"] = mc

    json.dump(out, open(out_path, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "coverage"}, indent=1))
    for p in cov:
        print(f"COV {p['category']:<15} {p['condition']:<18} n={p['n']:4d} cov={p['coverage']:.2f} "
              f"acc={p['accuracy']:.3f} kept={p['kept_vs_full']:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
