"""HD-EPIC VQA items (CVPR 2025; github.com/hd-epic/hd-epic-annotations).

We use the short-segment question types (a few seconds of one video). Each has
5 choices. Categories follow docs/task_taxonomy.md.

Oracle evidence: HD-EPIC annotates every object movement with a box at the
pick-up and at the put-down frame (scene-and-object-movements/*.json). For a
question window we take the union of the boxes whose frame falls inside the
window (+-0.5 s) as the evidence region, held fixed over the window. Items
without such a box have no oracle.

Measured gaze: Aria MPS eye gaze, projected into the MP4 frames beforehand by
scripts/prep_hdepic_gaze.py (one CSV per video, 1408-px frames; `gaze_orientation` picks
the raw or upright projection, checked by eye against the MP4).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.captaincook4d import LETTERS, FrameCache, Item, Segment

NATIVE = 1408  # HD-EPIC MP4 frames are 1408 x 1408

PROMPT = "{question}\n{options}\nAnswer with the letter of the correct option only."

# docs/task_taxonomy.md (HD-EPIC section, fixed before the first HD-EPIC run)
CATEGORY = {
    "fine_grained_action_recognition": "local-evidence",
    "fine_grained_how_recognition": "local-evidence",
    "fine_grained_why_recognition": "temporal-only",
    "gaze_gaze_estimation": "global",
    "gaze_interaction_anticipation": "local-evidence",
}


def _ts(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


class HDEpicItem(Item):
    question: str = ""

    def prompt(self) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(self.options))
        return PROMPT.format(question=self.question, options=opts)


class HDEpicFrames(FrameCache):
    def video_path(self, recording_id: str) -> str:
        return str(self.root / "Videos" / recording_id.split("-")[0] / f"{recording_id}.mp4")


class HDEpic:
    """Dataset interface used by scripts/run_experiment.py."""

    def __init__(self, d: dict, sampling: dict, full_hw: tuple[int, int], seed: int):
        self.d, self.sampling, self.full_hw = d, sampling, full_hw
        root = Path(d["root"])
        ann = root / "annotations"
        masks = json.load(open(ann / "scene-and-object-movements/mask_info.json"))
        assoc = json.load(open(ann / "scene-and-object-movements/assoc_info.json"))
        # video -> list of (time_s, bbox in native pixels)
        self.events = defaultdict(list)
        for vid, objs in assoc.items():
            for o in objs.values():
                for tr in o["tracks"]:
                    for mk in tr["masks"]:
                        m = masks[vid].get(mk)
                        if m:
                            self.events[vid].append((m["frame_number"] / 30.0, m["bbox"]))
        have_video = {p.stem for p in (root / "Videos").glob("*/*.mp4")}

        rng = random.Random(seed)
        self.segments, self.items, segs = [], [], {}
        for task, spec in d["tasks"].items():
            qs = json.load(open(ann / "vqa-benchmark" / f"{task}.json"))
            pool = []
            for qid, q in sorted(qs.items()):
                v = q["inputs"].get("video 1", {})
                if "start_time" not in v or v["id"] not in have_video or len(q["inputs"]) != 1:
                    continue
                t0, t1 = _ts(v["start_time"]), _ts(v["end_time"])
                if spec.get("require_oracle") and self._oracle_raw(v["id"], t0, t1) is None:
                    continue
                pool.append((qid, q, v["id"], t0, t1))
            rng.shuffle(pool)
            for qid, q, vid, t0, t1 in pool[:spec["n"]]:
                key = f"{vid}_{t0:.3f}_{t1:.3f}"
                if key not in segs:
                    segs[key] = Segment(vid, key, t0, t1, "", [])
                    self.segments.append(segs[key])
                it = HDEpicItem(f"{task}_{qid}", task, segs[key], list(q["choices"]), LETTERS[q["correct_idx"]],
                                [CATEGORY[task]], [q["question"]] + list(q["choices"]))
                it.question = q["question"]
                self.items.append(it)
        self.cache = HDEpicFrames(d["root"], "Videos", ".mp4", d["frame_cache"], full_hw)

    # --- frames -------------------------------------------------------------------------------
    def times(self, seg: Segment) -> np.ndarray:
        """A fixed number of frames per segment (segments are 1 to 10 s long)."""
        n = self.sampling["frames_per_segment"]
        return np.linspace(seg.t_start, seg.t_end, n, endpoint=False) + (seg.t_end - seg.t_start) / (2 * n)

    def prepare_frames(self, segs: list[Segment]) -> None:
        self.cache.prepare_all([(s, self.times(s)) for s in segs], workers=self.d.get("decode_workers", 8))

    def load_frames(self, seg: Segment) -> np.ndarray:
        return self.cache.load(seg, self.times(seg))

    def gaze_video(self, seg: Segment) -> str:
        return self.cache.video_path(seg.recording_id)

    # --- pointing signals ---------------------------------------------------------------------
    def _scale(self) -> np.ndarray:
        H, W = self.full_hw
        return np.array([W / NATIVE, H / NATIVE, W / NATIVE, H / NATIVE])

    def _oracle_raw(self, vid: str, t0: float, t1: float):
        boxes = [b for t, b in self.events.get(vid, []) if t0 - 0.5 <= t <= t1 + 0.5]
        if not boxes:
            return None
        b = np.array(boxes)
        return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])

    def oracle_boxes(self, item: Item) -> np.ndarray | None:
        s = item.segment
        raw = self._oracle_raw(s.recording_id, s.t_start, s.t_end)
        if raw is None:
            return None
        return np.tile(raw * self._scale(), (len(self.times(s)), 1))

    def measured_gaze(self, seg: Segment) -> np.ndarray | None:
        p = Path(self.d["gaze_dir"]) / f"{seg.recording_id}.csv"
        if not p.exists():
            return None
        g = np.loadtxt(p, delimiter=",", skiprows=1)  # time_s, x_raw, y_raw, x_up, y_up (native px)
        cols = [0, 3, 4] if self.d.get("gaze_orientation", "up") == "up" else [0, 1, 2]
        g = g[:, cols]
        g = g[np.isfinite(g).all(axis=1)]
        if len(g) == 0:
            return None
        t = self.times(seg)
        x = np.interp(t, g[:, 0], g[:, 1])
        y = np.interp(t, g[:, 0], g[:, 2])
        H, W = self.full_hw
        return np.stack([np.clip(x * W / NATIVE, 0, W - 1), np.clip(y * H / NATIVE, 0, H - 1)], axis=1)
