"""Turn per-item rows into per-(task, category, condition) scores.

Can be re-run offline: python -m eval.summarize <items.jsonl> <config.yaml>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import yaml

from eval.stats import bootstrap_ci, paired_diff_ci


def summarize(rows: list[dict], cfg: dict) -> dict:
    conds = cfg["conditions"]
    full = next(c["name"] for c in conds if c["view"] == "full")
    base_of = {c["name"]: next((b["name"] for b in conds if b["view"] == "downsample"
                                and b.get("budget") == c.get("budget")), None) for c in conds}
    seed = cfg["seed"]
    out = {}
    for task in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == task]
        # Item groups: whole task, then each taxonomy category.
        groups = {"all": None}
        for r in tr:
            for c in r["categories"]:
                groups[c] = c
        res_task = {}
        for gname, cat in groups.items():
            gr = [r for r in tr if cat is None or cat in r["categories"]]
            by = defaultdict(dict)
            for r in gr:
                by[r["condition"]][r["item_id"]] = r
            ids_full = sorted(by[full])
            res = {"n_items": len(ids_full), "conditions": {}}
            for c in conds:
                n = c["name"]
                ids = [i for i in ids_full if i in by[n]]  # a signal can be missing for some items
                if not ids:
                    continue
                corr = np.array([by[n][i]["correct"] for i in ids], float)
                corr_full = np.array([by[full][i]["correct"] for i in ids], float)
                rs = [by[n][i] for i in ids]
                lo, hi = bootstrap_ci(corr, seed=seed)
                s = {"n_items": len(ids), "accuracy": float(corr.mean()), "ci95": [lo, hi],
                     "kept_vs_full": float(corr.mean() / max(corr_full.mean(), 1e-9)),
                     "visual_tokens_mean": float(np.mean([r["visual_tokens"] for r in rs])),
                     "token_share_vs_full": float(sum(r["visual_tokens"] for r in rs) /
                                                  max(sum(by[full][i]["visual_tokens"] for i in ids), 1)),
                     "latency_s_mean": float(np.mean([r["latency_s"] for r in rs])),
                     "unparsed": int(sum(r["pred"] is None for r in rs))}
                if task == "error_detection" and gname == "all":
                    pos = np.array([by[n][i]["answer"] == "A" for i in ids])
                    if pos.any() and (~pos).any():
                        s["recall_errors"] = float(corr[pos].mean())
                        s["specificity"] = float(corr[~pos].mean())
                        s["balanced_accuracy"] = (s["recall_errors"] + s["specificity"]) / 2
                b = base_of[n]
                if b and b != n and c["view"] != "full":
                    both = [i for i in ids if i in by[b]]
                    s["diff_vs_downsample_same_budget"] = paired_diff_ci(
                        np.array([by[n][i]["correct"] for i in both], float),
                        np.array([by[b][i]["correct"] for i in both], float), seed=seed)
                res["conditions"][n] = s
            res_task[gname] = res
        out[task] = res_task
    return out


def print_table(summary: dict) -> None:
    for task, groups in summary.items():
        for g, res in groups.items():
            print(f"TABLE task={task} group={g} n={res['n_items']}")
            for n, s in res["conditions"].items():
                extra = f" bal={s['balanced_accuracy']:.3f}" if "balanced_accuracy" in s else ""
                d = s.get("diff_vs_downsample_same_budget")
                dd = f" vs_down={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else ""
                print(f"  {n:<22} n={s['n_items']:4d} acc={s['accuracy']:.3f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}] "
                      f"tok={s['token_share_vs_full']:.3f}{extra}{dd}")


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    print_table(summarize(rows, yaml.safe_load(open(sys.argv[2]))))
