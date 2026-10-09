"""Render worked examples for the experiment log: what the model saw, and what it said.

Usage (CPU batch job):
  python scripts/make_examples.py <items.jsonl> <frame_cache_dir> <hdepic_vqa_dir> <out_dir> <item_id> [<item_id> ...]

For each item: one PNG with the middle frame (crop windows and the oracle box
drawn on it) and the exact pictures fed to the model for each condition at the
10% budget, plus a JSON with the question, options, answer and every reply.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

COLOURS = {"oracle": (220, 40, 40), "centre": (240, 200, 0), "gaze": (40, 200, 40),
           "object": (40, 120, 255), "gazeobj": (200, 60, 220), "random": (150, 150, 150)}
GRID = {"b10": (11, 11), "b25": (18, 18)}
PERIPHERY = {"b10": ((5, 5), (8, 12)), "b25": ((8, 8), (13, 20))}


def crop_like_model(frame, box, grid):
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = grid[0] * 28, grid[1] * 28
    return cv2.resize(frame[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)


def tile(img, size=300):
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST)
    out = np.full((size, size, 3), 255, np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main(items_path, cache, vqa_dir, out_dir, *item_ids):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by = defaultdict(dict)
    for line in open(items_path):
        r = json.loads(line)
        if r["item_id"] in item_ids:
            by[r["item_id"]][r["condition"]] = r
    for iid, cs in by.items():
        any_row = next(iter(cs.values()))
        seg_dir = Path(cache) / any_row["segment"]
        paths = sorted(seg_dir.glob("*.jpg"), key=lambda p: float(p.stem))
        frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        mid = len(frames) // 2
        f = frames[mid]
        # Frame with boxes drawn
        drawn = f.copy()
        if "oracle_box" in any_row:
            b = [int(v) for v in any_row["oracle_box"]]
            cv2.rectangle(drawn, (b[0], b[1]), (b[2], b[3]), COLOURS["oracle"], 4)
        panels = [label(tile(drawn), "frame (red = oracle box)")]
        for name in ["full", "down_b10", "oracle_b10", "centre_b10", "gaze_b10", "object_b10", "oracle_per_b10"]:
            if name not in cs:
                continue
            r = cs[name]
            if name == "full":
                img = f
            elif name.startswith("down"):
                g = GRID[r["budget"]]
                img = cv2.resize(f, (g[1] * 28, g[0] * 28), interpolation=cv2.INTER_AREA)
            elif "_per_" in name:
                pg, cg = PERIPHERY[r["budget"]]
                per = cv2.resize(f, (pg[1] * 28, pg[0] * 28), interpolation=cv2.INTER_AREA)
                crop = crop_like_model(f, r["boxes"][mid], cg)
                img = np.full((max(per.shape[0], crop.shape[0]), per.shape[1] + 10 + crop.shape[1], 3), 255, np.uint8)
                img[:per.shape[0], :per.shape[1]] = per
                img[:crop.shape[0], per.shape[1] + 10:] = crop
            else:
                img = crop_like_model(f, r["boxes"][mid], GRID[r["budget"]])
                b = [int(v) for v in r["boxes"][mid]]
                cv2.rectangle(drawn, (b[0], b[1]), (b[2], b[3]), COLOURS.get(name.split("_")[0], (0, 0, 0)), 3)
            hw = r["frame_hw"]
            panels.append(label(tile(img), f"{name}: {'+'.join(f'{h}x{w}' for h, w in hw)} px, "
                                           f"{r['visual_tokens']} tok -> {r['pred']}"))
        panels[0] = label(tile(drawn), "frame + crop windows")
        while len(panels) % 4:
            panels.append(np.full_like(panels[0], 255))
        rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
        cv2.imwrite(str(out / f"{iid}.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
        # Question text from the benchmark file
        task = any_row["task"]
        q = {}
        if vqa_dir != "-":
            qs = json.load(open(Path(vqa_dir) / f"{task}.json"))
            q = qs.get(iid[len(task) + 1:], {})
        json.dump({"item_id": iid, "task": task, "question": q.get("question"), "choices": q.get("choices"),
                   "answer": any_row["answer"], "n_frames": any_row["n_frames"],
                   "replies": {n: {"reply": r["reply"], "correct": r["correct"], "visual_tokens": r["visual_tokens"]}
                               for n, r in cs.items()}},
                  open(out / f"{iid}.json", "w"), indent=1)
        print("wrote", iid)


if __name__ == "__main__":
    main(*sys.argv[1:])
