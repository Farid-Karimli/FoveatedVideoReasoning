"""Object pointing signal: Grounding DINO boxes for objects named in the text.

Object phrases come from the step descriptions (CaptainCook4D gives no object
labels). Per frame we keep the single highest-scoring box. Frames with no box
above threshold reuse the previous frame's box, or a centre point if there is none.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Words that never name an object in CaptainCook4D step text.
_STOP = {
    "a", "an", "the", "of", "and", "or", "it", "them", "its", "their", "this", "that", "for", "about", "until",
    "all", "some", "each", "other", "remaining", "together", "well", "evenly", "into", "onto", "on", "in", "to",
    "with", "from", "over", "at", "by", "up", "out", "off", "inside", "after", "before", "then", "is", "are",
    "minutes", "minute", "seconds", "second", "min", "sec", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
    "tbsp", "tsp", "cup", "cups", "oz", "ounce", "ounces", "inch", "inches", "pinch", "piece", "pieces", "half",
    "small", "large", "medium", "few", "more", "high", "low", "power", "heat", "time", "times", "x",
}
_SPLIT = re.compile(r"\b(?:and|with|on|in|into|onto|to|from|over|of|for|using|at)\b|[,;()/]")


def object_phrases(description: str, max_words: int = 3) -> list[str]:
    """'Chop 1 garlic clove on a cutting board' -> ['garlic clove', 'cutting board'].

    Drops the leading verb (CaptainCook4D descriptions start with one), numbers,
    units and stop words, then splits on prepositions.
    """
    text = description.lower()
    words = text.split()
    text = " ".join(words[1:]) if words else text  # leading verb
    out = []
    for chunk in _SPLIT.split(text):
        toks = [t for t in re.findall(r"[a-z]+", chunk) if t not in _STOP and len(t) > 1]
        if toks:
            out.append(" ".join(toks[-max_words:]))
    return list(dict.fromkeys(out))


class ObjectDetector:
    def __init__(self, model_id: str, device: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device, self.box_t, self.text_t = device, box_threshold, text_threshold

    @torch.no_grad()
    def best_boxes(self, frames: np.ndarray, phrases: list[str], batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """One (x0, y0, x1, y1) box per frame and a found-flag per frame."""
        H, W = frames.shape[1:3]
        prompt = " . ".join(phrases) + " ."
        boxes, found = [], []
        for i in range(0, len(frames), batch):
            chunk = [Image.fromarray(f) for f in frames[i:i + batch]]
            inputs = self.processor(images=chunk, text=[prompt] * len(chunk), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.box_t, text_threshold=self.text_t,
                target_sizes=[(H, W)] * len(chunk))
            for r in res:
                if len(r["scores"]):
                    boxes.append(r["boxes"][int(r["scores"].argmax())].float().cpu().numpy())
                    found.append(True)
                else:
                    boxes.append(None)
                    found.append(False)
        # Fill frames without a detection: previous box, else next box, else centre point.
        filled, last = [], None
        for b in boxes:
            last = b if b is not None else last
            filled.append(last)
        nxt = next((b for b in boxes if b is not None), np.array([W / 2, H / 2, W / 2, H / 2], dtype=np.float32))
        filled = [b if b is not None else nxt for b in filled]
        return np.stack(filled), np.array(found)


def gaze_object_boxes(gaze_xy: np.ndarray, obj_boxes: np.ndarray) -> np.ndarray:
    """Gaze + object: the object box if gaze falls inside it, else the box spanning both."""
    out = []
    for (gx, gy), (x0, y0, x1, y1) in zip(gaze_xy, obj_boxes):
        if x0 <= gx <= x1 and y0 <= gy <= y1:
            out.append((x0, y0, x1, y1))
        else:
            out.append((min(x0, gx), min(y0, gy), max(x1, gx), max(y1, gy)))
    return np.array(out, dtype=np.float32)
