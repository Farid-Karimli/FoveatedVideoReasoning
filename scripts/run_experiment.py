"""Run one experiment config: every item through every condition on a frozen VLM.

Usage: python scripts/run_experiment.py configs/experiment.yaml

Writes per-item JSONL and a summary JSON under <out_root>/<run_tag>/, and prints
the summary to stdout (lines starting with "SUMMARY").
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# The SCC login profile sets TRANSFORMERS_CACHE to another project's folder, and
# transformers prefers it over HF_HOME. Point every HF cache at HF_HOME/hub
# before importing transformers.
if "HF_HOME" in os.environ:
    for var in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ[var] = os.path.join(os.environ["HF_HOME"], "hub")

import numpy as np
import torch
import yaml

from eval.captaincook4d import LETTERS, build_items, load_frames, sample_times
from eval.qwen import QwenVL, parse_letter
from eval.stats import bootstrap_ci, paired_diff_ci
from fovea import views


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown (orx run " + os.environ.get("ORX_RUN_ID", "?") + ")"


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda"
    full_hw = views.grid_px(tuple(cfg["full_grid"]))
    budgets = {name: tuple(g) for name, g in cfg["budgets"].items()}

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg["name"]
    out_dir = Path(cfg["out_root"]) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_tag={run_tag} out_dir={out_dir} commit={git_commit()}", flush=True)

    d = cfg["dataset"]
    items = build_items(d["root"], d["video_dir"], d["n_items"], d["n_options"], d["min_duration_s"], cfg["seed"])
    print(f"items={len(items)} full_hw={full_hw} budgets={budgets}", flush=True)

    vlm = QwenVL(cfg["model"]["id"])
    gaze_model = None
    if any(c["signal"] == "gaze_predicted" for c in cfg["conditions"]):
        from fovea.gaze_egogazelite import load_gaze_model, predict_gaze
        gaze_model = load_gaze_model(cfg["gaze"]["checkpoint"], device)

    rows = []
    f_items = open(out_dir / "items.jsonl", "w")
    for k, it in enumerate(items):
        times = sample_times(it, cfg["sampling"]["fps"], cfg["sampling"]["max_frames"])
        frames = load_frames(it.video_path, times, full_hw)
        points = {"centre": views.centre_points(len(frames), full_hw)}
        if gaze_model is not None:
            t0 = time.perf_counter()
            points["gaze_predicted"] = predict_gaze(gaze_model, it.video_path, it.t_start, times, full_hw, device)
            gaze_s = time.perf_counter() - t0
        for c in cfg["conditions"]:
            boxes = None
            if c["view"] == "full":
                x = views.full_frames(frames)
            elif c["view"] == "downsample":
                x = views.downsample(frames, budgets[c["budget"]])
            elif c["view"] == "crop":
                x, boxes = views.crop_at(frames, points[c["signal"]], budgets[c["budget"]])
            else:
                raise ValueError(c["view"])
            expected = views.expected_tokens(len(x), x.shape[1:3])
            r = vlm.answer(x, it.prompt(), fps=cfg["sampling"]["fps"])
            # Hard check: the model must see exactly the tokens our grid implies.
            assert r["visual_tokens"] == expected, (c["name"], r["visual_tokens"], expected)
            pred = parse_letter(r["reply"], LETTERS[:len(it.options)])
            row = {"item_id": it.item_id, "condition": c["name"], "budget": c.get("budget", "full"),
                   "n_frames": len(x), "frame_hw": list(x.shape[1:3]), "visual_tokens": r["visual_tokens"],
                   "total_tokens": r["total_tokens"], "latency_s": round(r["latency_s"], 4),
                   "reply": r["reply"], "pred": pred, "answer": it.answer, "correct": pred == it.answer,
                   "has_errors": it.has_errors, "t_start": it.t_start, "t_end": it.t_end}
            if boxes is not None:
                row["boxes"] = boxes.tolist()
            if c.get("signal") == "gaze_predicted":
                row["gaze_latency_s"] = round(gaze_s, 4)
            rows.append(row)
            f_items.write(json.dumps(row) + "\n")
        f_items.flush()
        print(f"[{k + 1}/{len(items)}] {it.item_id} " +
              " ".join(f"{r['condition']}={int(r['correct'])}/{r['visual_tokens']}tok"
                       for r in rows[-len(cfg['conditions']):]), flush=True)
    f_items.close()

    # Summary: accuracy with bootstrap CI, measured tokens and latency, paired diff vs downsample at same budget.
    by = {c["name"]: [r for r in rows if r["condition"] == c["name"]] for c in cfg["conditions"]}
    acc = {n: np.array([r["correct"] for r in rs], dtype=float) for n, rs in by.items()}
    full_name = next(c["name"] for c in cfg["conditions"] if c["view"] == "full")
    summary = {"run_tag": run_tag, "commit": git_commit(), "config": cfg, "n_items": len(items), "conditions": {}}
    for c in cfg["conditions"]:
        n, rs = c["name"], by[c["name"]]
        lo, hi = bootstrap_ci(acc[n], seed=cfg["seed"])
        tok = np.array([r["visual_tokens"] for r in rs])
        s = {"accuracy": float(acc[n].mean()), "ci95": [lo, hi],
             "kept_vs_full": float(acc[n].mean() / max(acc[full_name].mean(), 1e-9)),
             "visual_tokens_mean": float(tok.mean()), "visual_tokens_min": int(tok.min()),
             "visual_tokens_max": int(tok.max()),
             "token_share_vs_full": float(tok.sum() / sum(r["visual_tokens"] for r in by[full_name])),
             "latency_s_mean": float(np.mean([r["latency_s"] for r in rs])),
             "unparsed": int(sum(r["pred"] is None for r in rs))}
        base = next((b["name"] for b in cfg["conditions"]
                     if b["view"] == "downsample" and b.get("budget") == c.get("budget")), None)
        if c["view"] == "crop" and base:
            s["diff_vs_downsample_same_budget"] = paired_diff_ci(acc[n], acc[base], seed=cfg["seed"])
        summary["conditions"][n] = s
        print("SUMMARY " + json.dumps({"condition": n, **s}), flush=True)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"SUMMARY_PATH {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
