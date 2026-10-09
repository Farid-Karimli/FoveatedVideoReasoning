"""CaptainCook4D items for two of our own tasks (the dataset has no question benchmark).

Each annotated step segment gives two items:
- step recognition: which step is this? true step + distractor steps of the same recipe.
- error detection: given the step text, did the person make a mistake? (A. Yes / B. No)

Categories follow docs/task_taxonomy.md.
"""

from __future__ import annotations

import csv
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LETTERS = "ABCDE"

PROMPTS = {
    "step_recognition": (
        "This video shows one step of a person cooking a recipe.\n"
        "Which step is the person performing?\n"
        "{options}\n"
        "Answer with the letter of the correct option only."
    ),
    "error_detection": (
        "This video shows a person performing this cooking step: \"{step}\".\n"
        "Did the person make a mistake while performing this step?\n"
        "{options}\n"
        "Answer with the letter of the correct option only."
    ),
}

# docs/task_taxonomy.md (fixed before step 3)
ERROR_CATEGORY = {
    "Technique Error": "local-evidence", "Measurement Error": "local-evidence",
    "Preparation Error": "local-evidence", "Temperature Error": "off-attention",
    "Order Error": "temporal-only", "Timing Error": "temporal-only", "Missing Step": "temporal-only",
    "Other": "unassigned",
}


@dataclass
class Segment:
    recording_id: str
    step_id: str
    t_start: float
    t_end: float
    step_text: str
    error_types: list[str]

    @property
    def seg_id(self) -> str:
        return f"{self.recording_id}_s{self.step_id}_{self.t_start:.1f}"


@dataclass
class Item:
    item_id: str
    task: str
    segment: Segment
    options: list[str]
    answer: str  # letter
    categories: list[str] = field(default_factory=list)  # taxonomy categories ([] = none)
    object_text: list[str] = field(default_factory=list)  # texts whose objects the object signal looks for

    def prompt(self) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(self.options))
        return PROMPTS[self.task].format(options=opts, step=self.segment.step_text)


def _clean(desc: str) -> str:
    """'Peel-Peel 1 garlic clove' -> 'Peel 1 garlic clove'."""
    return desc.split("-", 1)[1].strip() if "-" in desc else desc.strip()


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return -1.0


def load_segments(root: str, recordings: list[str], min_duration: float) -> list[Segment]:
    rows = list(csv.reader(open(Path(root) / "annotations/annotation_csv/error_annotations.csv")))
    head = rows[0]
    known = {r[0] for r in rows[1:]}
    bad = [r for r in recordings if not isinstance(r, str) or r not in known]
    if bad:
        raise ValueError(f"unknown recording ids (quote them in YAML): {bad}")
    type_cols = list(range(6, len(head), 2))
    segs = []
    for r in rows[1:]:
        if r[0] not in recordings:
            continue
        t0, t1 = _float(r[2]), _float(r[3])
        if t0 < 0 or t1 - t0 < min_duration:
            continue
        types = [head[i] for i in type_cols if r[i] == "1"]
        segs.append(Segment(r[0], r[1], t0, t1, _clean(r[4]), types if r[5] == "True" else []))
    if not segs:
        raise ValueError("no step segments selected")
    return segs


def build_items(root: str, recordings: list[str], tasks: list[str], n_options: int, min_duration: float,
                seed: int) -> tuple[list[Segment], list[Item]]:
    segs = load_segments(root, recordings, min_duration)
    act_steps: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(open(Path(root) / "annotations/annotation_csv/activity_step_description.csv")):
        act_steps.setdefault(r["activity_idx"], {})[r["step_index"]] = _clean(r["step_description"])
    rng = random.Random(seed)
    items = []
    for s in segs:
        if "step_recognition" in tasks:
            steps = act_steps[s.recording_id.split("_")[0]]
            others = sorted({d for k, d in steps.items() if k != s.step_id and d != s.step_text})
            opts = rng.sample(others, n_options - 1) + [s.step_text]
            rng.shuffle(opts)
            # Object signal may only use what the question names: all options.
            items.append(Item(f"{s.seg_id}_rec", "step_recognition", s, opts, LETTERS[opts.index(s.step_text)],
                              ["local-evidence"], list(opts)))
        if "error_detection" in tasks:
            cats = sorted({ERROR_CATEGORY.get(t, "unassigned") for t in s.error_types})
            items.append(Item(f"{s.seg_id}_err", "error_detection", s, ["Yes", "No"],
                              "A" if s.error_types else "B", cats, [s.step_text]))
    return segs, items


def sample_times(seg: Segment, fps: float, max_frames: int) -> np.ndarray:
    """Frame times at `fps` (bin centres), capped and made even for Qwen's 2-frame tokens."""
    n = max(2, int((seg.t_end - seg.t_start) * fps))
    n = min(n, max_frames)
    n -= n % 2
    return np.linspace(seg.t_start, seg.t_end, n, endpoint=False) + (seg.t_end - seg.t_start) / (2 * n)


def _decode(video_path: str, times: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Decode RGB frames at the given times with the ffmpeg binary, one process per frame.

    OpenCV's in-process decoder got jobs killed for memory on some 4K HEVC files
    (exit 137 at the same spot three times). A separate ffmpeg process per frame
    keeps memory bounded and turns a bad spot into a clear error for that frame.
    """
    import subprocess

    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = out_hw
    frames = []
    for t in times:
        cmd = [exe, "-v", "error", "-threads", "2", "-ss", f"{float(t):.3f}", "-i", video_path,
               "-frames:v", "1", "-vf", f"scale={W}:{H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        if out.returncode != 0 or len(out.stdout) != H * W * 3:
            raise RuntimeError(f"ffmpeg failed on {video_path} at {t:.2f}s: {out.stderr.decode()[-300:]}")
        frames.append(np.frombuffer(out.stdout, np.uint8).reshape(H, W, 3))
    return np.stack(frames)


class FrameCache:
    """Decode sampled frames from the 4K files once, keep them as JPEGs on /projectnb.

    4K HEVC decodes at ~12 frames/s and each seek costs ~0.6 s, so decoding inside
    the condition loop would dominate run time. JPEG (quality 95) is applied to
    every condition alike.
    """

    def __init__(self, root: str, video_dir: str, suffix: str, cache_dir: str, out_hw: tuple[int, int]):
        self.root, self.video_dir, self.suffix = Path(root), video_dir, suffix
        self.cache = Path(cache_dir) / f"{out_hw[0]}x{out_hw[1]}"
        self.out_hw = out_hw

    def video_path(self, recording_id: str) -> str:
        return str(self.root / self.video_dir / f"{recording_id}{self.suffix}")

    def _paths(self, seg: Segment, times: np.ndarray) -> list[Path]:
        return [self.cache / seg.seg_id / f"{t:.3f}.jpg" for t in times]

    def prepare(self, seg: Segment, times: np.ndarray) -> None:
        paths = self._paths(seg, times)
        if all(p.exists() for p in paths):
            return
        frames = _decode(self.video_path(seg.recording_id), times, self.out_hw)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for p, f in zip(paths, frames):
            cv2.imwrite(str(p), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    def prepare_all(self, jobs: list[tuple[Segment, np.ndarray]], workers: int) -> None:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda j: self.prepare(*j), jobs))

    def load(self, seg: Segment, times: np.ndarray) -> np.ndarray:
        return np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in self._paths(seg, times)])


class CaptainCook4D:
    """Dataset interface used by scripts/run_experiment.py."""

    def __init__(self, d: dict, sampling: dict, full_hw: tuple[int, int], seed: int):
        self.segments, self.items = build_items(d["root"], d["recordings"], d["tasks"], d["n_options"],
                                                d["min_duration_s"], seed)
        self.sampling, self.full_hw, self.d = sampling, full_hw, d
        self.cache = FrameCache(d["root"], d["video_dir"], d["video_suffix"], d["frame_cache"], full_hw)

    def times(self, seg: Segment) -> np.ndarray:
        return sample_times(seg, self.sampling["fps"], self.sampling["max_frames"])

    def prepare_frames(self, segs: list[Segment]) -> None:
        self.cache.prepare_all([(s, self.times(s)) for s in segs], workers=self.d.get("decode_workers", 8))

    def load_frames(self, seg: Segment) -> np.ndarray:
        return self.cache.load(seg, self.times(seg))

    def gaze_video(self, seg: Segment) -> str:
        # Gaze runs on the 360p copy (same frames and timing; model input is 300x300 anyway).
        return self.d["gaze_video"].format(rec=seg.recording_id)

    def measured_gaze(self, seg: Segment) -> np.ndarray | None:
        return None  # CaptainCook4D has no eye tracking (docs/open_questions.md)

    def oracle_boxes(self, item: Item) -> np.ndarray | None:
        return None  # no evidence annotation
