"""CaptainCook4D step-recognition items.

CaptainCook4D has no question-answer benchmark. For the smoke test we build a
multiple-choice task from its step annotations: show one annotated step segment,
ask which step it is, with the true step plus distractors drawn from the other
steps of the same recipe. Category: hand-object interaction, which our taxonomy
expects to be local-evidence (to be fixed in docs/task_taxonomy.md before step 3).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LETTERS = "ABCDE"

PROMPT = (
    "This video shows one step of a person cooking a recipe.\n"
    "Which step is the person performing?\n"
    "{options}\n"
    "Answer with the letter of the correct option only."
)


@dataclass
class Item:
    item_id: str
    recording_id: str
    video_path: str
    t_start: float
    t_end: float
    step_id: str
    options: list[str]
    answer: str  # letter
    has_errors: bool

    def prompt(self) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(self.options))
        return PROMPT.format(options=opts)


def _clean(desc: str) -> str:
    """'Peel-Peel 1 garlic clove' -> 'Peel 1 garlic clove'."""
    return desc.split("-", 1)[1].strip() if "-" in desc else desc.strip()


def build_items(root: str, video_dir: str, n_items: int, n_options: int, min_duration: float,
                seed: int) -> list[Item]:
    """Sample n_items step segments from recordings whose video is on disk."""
    root = Path(root)
    videos = {p.name.split("_360p")[0]: str(p) for p in (root / video_dir).glob("*.mp4")}
    rows = list(csv.DictReader(open(root / "annotations/annotation_csv/step_annotations.csv")))
    # recording "4_2" -> activity 4; all steps of that activity are candidate distractors
    act_steps: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(open(root / "annotations/annotation_csv/activity_step_description.csv")):
        act_steps.setdefault(r["activity_idx"], {})[r["step_index"]] = _clean(r["step_description"])

    pool = [r for r in rows if r["recording_id"] in videos
            and float(r["end_time"]) - float(r["start_time"]) >= min_duration]
    rng = random.Random(seed)
    rng.shuffle(pool)
    items = []
    for r in pool[:n_items]:
        steps = act_steps[r["recording_id"].split("_")[0]]
        true = _clean(r["description"])
        others = sorted({d for s, d in steps.items() if s != r["step_id"] and d != true})
        opts = rng.sample(others, n_options - 1) + [true]
        rng.shuffle(opts)
        items.append(Item(
            item_id=f"{r['recording_id']}_s{r['step_id']}_{float(r['start_time']):.1f}",
            recording_id=r["recording_id"], video_path=videos[r["recording_id"]],
            t_start=float(r["start_time"]), t_end=float(r["end_time"]), step_id=r["step_id"],
            options=opts, answer=LETTERS[opts.index(true)], has_errors=r["has_errors"] == "True",
        ))
    if len(items) < n_items:
        raise RuntimeError(f"only {len(items)} eligible items, wanted {n_items}")
    return items


def sample_times(item: Item, fps: float, max_frames: int) -> np.ndarray:
    """Frame times at `fps` (bin centres), capped and made even for Qwen's 2-frame tokens."""
    n = max(2, int((item.t_end - item.t_start) * fps))
    n = min(n, max_frames)
    n -= n % 2
    return np.linspace(item.t_start, item.t_end, n, endpoint=False) + (item.t_end - item.t_start) / (2 * n)


def load_frames(video_path: str, times: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Decode RGB frames at the given times, resized to out_hw (H, W)."""
    cap = cv2.VideoCapture(video_path)
    H, W = out_hw
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {video_path} at {t:.2f}s")
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames)
