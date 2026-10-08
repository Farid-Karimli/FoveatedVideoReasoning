"""Run one experiment config: every item through every condition on a frozen VLM.

Usage: python scripts/run_experiment.py configs/experiment.yaml

Writes per-item JSONL and a summary JSON under <out_root>/<run_tag>/, and prints
the summary to stdout (lines starting with "TABLE" / "SUMMARY").
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from collections import defaultdict
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

from eval.captaincook4d import LETTERS, CaptainCook4D
from eval.qwen import QwenVL, parse_letter
from eval.summarize import print_table, summarize
from fovea import views


def run_id() -> str:
    return os.environ.get("ORX_RUN_ID", "local")


def build_view(c: dict, frames: np.ndarray, boxes: dict, cfg: dict) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Frame stacks the VLM sees for one condition, plus the crop boxes used."""
    budgets = cfg["budgets"]
    if c["view"] == "full":
        return [frames], None
    if c["view"] == "downsample":
        return [views.downsample(frames, tuple(budgets[c["budget"]]))], None
    if c["view"] == "crop":
        grid = tuple(budgets[c["budget"]])
        x, used = views.crop_regions(frames, boxes[c["signal"]](grid), grid)
        return [x], used
    if c["view"] == "crop_periphery":
        split = cfg["periphery_splits"][c["budget"]]
        grid = tuple(split["crop"])
        x, used = views.crop_regions(frames, boxes[c["signal"]](grid), grid)
        return [views.downsample(frames, tuple(split["periphery"])), x], used
    raise ValueError(c["view"])


DATASETS = {"captaincook4d": CaptainCook4D}


def load_dataset(d: dict, sampling: dict, full_hw: tuple[int, int], seed: int):
    if d["name"] == "hdepic":
        from eval.hdepic import HDEpic
        return HDEpic(d, sampling, full_hw, seed)
    return DATASETS[d["name"]](d, sampling, full_hw, seed)


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_hw = views.grid_px(tuple(cfg["full_grid"]))
    conds = cfg["conditions"]
    signals = {c.get("signal") for c in conds}

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + cfg["name"]
    out_dir = Path(cfg["out_root"]) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_tag={run_tag} out_dir={out_dir} orx_run={run_id()}", flush=True)

    d = cfg["dataset"]
    ds = load_dataset(d, cfg["sampling"], full_hw, cfg["seed"])
    segs, items = ds.segments, ds.items
    if d.get("max_segments"):
        keep = {s.seg_id for s in segs[:d["max_segments"]]}
        segs = [s for s in segs if s.seg_id in keep]
        items = [i for i in items if i.segment.seg_id in keep]
    seg_items = defaultdict(list)
    for it in items:
        seg_items[it.segment.seg_id].append(it)
    print(f"segments={len(segs)} items={len(items)} full_hw={full_hw}", flush=True)

    # Decode all sampled frames up front (threaded, cached on disk).
    t0 = time.perf_counter()
    ds.prepare_frames(segs)
    print(f"frames ready in {time.perf_counter() - t0:.0f}s", flush=True)

    vlm = QwenVL(cfg["model"]["id"])
    gaze_model = detector = None
    if signals & {"gaze_predicted"} or (signals & {"gaze_object"} and cfg.get("gaze_object_source", "gaze_predicted") == "gaze_predicted"):
        from fovea.gaze_egogazelite import load_gaze_model, predict_gaze
        gaze_model = load_gaze_model(cfg["gaze"]["checkpoint"], device)
    if signals & {"object", "gaze_object"}:
        from fovea.objects import ObjectDetector, gaze_object_boxes, object_phrases
        detector = ObjectDetector(cfg["detector"]["id"], device, cfg["detector"]["box_threshold"],
                                  cfg["detector"]["text_threshold"])

    rows = []
    f_items = open(out_dir / "items.jsonl", "w")
    for k, s in enumerate(segs):
        frames = ds.load_frames(s)
        tt = ds.times(s)
        n = len(frames)
        centre = views.points_to_boxes(views.centre_points(n, full_hw))
        seg_seed = zlib.crc32(s.seg_id.encode()) + cfg["seed"]
        seg_signals = {
            "centre": lambda g: centre,
            "random": lambda g: views.points_to_boxes(views.random_points(n, full_hw, g, seg_seed)),
        }
        extra = {}
        if gaze_model is not None:
            t1 = time.perf_counter()
            # Gaze runs on the 360p copy (same frames and timing; model input is 300x300 anyway).
            pred_gaze = predict_gaze(gaze_model, ds.gaze_video(s), s.t_start, tt, full_hw, device)
            extra["gaze_latency_s"] = round(time.perf_counter() - t1, 4)
            seg_signals["gaze_predicted"] = lambda g, b=views.points_to_boxes(pred_gaze): b
        meas_gaze = ds.measured_gaze(s) if "gaze_measured" in signals or \
            cfg.get("gaze_object_source") == "gaze_measured" else None
        if meas_gaze is not None:
            seg_signals["gaze_measured"] = lambda g, b=views.points_to_boxes(meas_gaze): b
        # Gaze used by the gaze+object signal: predicted by default, measured where configured.
        gaze = meas_gaze if cfg.get("gaze_object_source") == "gaze_measured" else \
            (pred_gaze if gaze_model is not None else None)

        for it in seg_items[s.seg_id]:
            sig = dict(seg_signals)
            item_extra = dict(extra)
            oracle = ds.oracle_boxes(it)
            if oracle is not None:
                sig["oracle"] = lambda g, b=oracle: b
            if detector is not None:
                t1 = time.perf_counter()
                phrases = list(dict.fromkeys(p for txt in it.object_text for p in object_phrases(txt)))
                obj, found = detector.best_boxes(frames, phrases)
                item_extra.update(object_latency_s=round(time.perf_counter() - t1, 4), object_phrases=phrases,
                                  object_found_share=float(found.mean()))
                sig["object"] = lambda g, obj=obj: obj
                if gaze is not None:
                    go = gaze_object_boxes(gaze, obj)
                    sig["gaze_object"] = lambda g, go=go: go
            for c in conds:
                if c.get("signal") and c["signal"] not in sig:
                    continue  # signal unavailable for this item (e.g. no gaze recorded); logged as missing
                vids, used = build_view(c, frames, sig, cfg)
                expected = sum(views.expected_tokens(len(v), v.shape[1:3]) for v in vids)
                r = vlm.answer(vids, it.prompt(), fps=cfg["sampling"]["fps"])
                # Hard check: the model must see exactly the tokens our grids imply.
                assert r["visual_tokens"] == expected, (c["name"], r["visual_tokens"], expected)
                pred = parse_letter(r["reply"], LETTERS[:len(it.options)])
                row = {"item_id": it.item_id, "task": it.task, "categories": it.categories,
                       "segment": s.seg_id, "condition": c["name"], "budget": c.get("budget", "full"),
                       "n_frames": n, "frame_hw": [list(v.shape[1:3]) for v in vids],
                       "visual_tokens": r["visual_tokens"], "total_tokens": r["total_tokens"],
                       "latency_s": round(r["latency_s"], 4), "reply": r["reply"], "pred": pred,
                       "answer": it.answer, "correct": pred == it.answer, "error_types": s.error_types}
                if used is not None:
                    row["boxes"] = used.tolist()
                if c.get("signal") in ("gaze_predicted", "gaze_object"):
                    row["gaze_latency_s"] = item_extra.get("gaze_latency_s")
                if c.get("signal") in ("object", "gaze_object"):
                    row.update({k2: item_extra[k2] for k2 in ("object_latency_s", "object_phrases", "object_found_share")})
                rows.append(row)
                f_items.write(json.dumps(row) + "\n")
            f_items.flush()
        done = [r for r in rows if r["segment"] == s.seg_id]
        if not done:
            continue
        print(f"[{k + 1}/{len(segs)}] {s.seg_id} correct {sum(r['correct'] for r in done)}/{len(done)} "
              f"tokens full={done[0]['visual_tokens']}", flush=True)
    f_items.close()

    summary = {"run_tag": run_tag, "orx_run": run_id(), "config": cfg, "n_segments": len(segs),
               "n_items": len(items), "results": summarize(rows, cfg)}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print_table(summary["results"])
    print(f"SUMMARY_PATH {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
