"""Qwen2.5-VL wrapper that measures visual tokens and latency per call.

Frames are passed with do_resize=False, so the model sees exactly the pixels we
built. Visual tokens are counted from the processed input ids (number of
<|video_pad|> tokens), not inferred from pixel counts.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenVL:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.device = device
        self.video_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    @torch.no_grad()
    def answer(self, frames: np.ndarray, prompt: str, fps: float, max_new_tokens: int = 8) -> dict:
        """Run one multiple-choice query. frames: (T, H, W, 3) uint8 RGB."""
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[list(frames)], do_resize=False, fps=[fps],
                                return_tensors="pt").to(self.device)
        n_visual = int((inputs["input_ids"] == self.video_pad_id).sum())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"reply": reply.strip(), "visual_tokens": n_visual,
                "total_tokens": int(inputs["input_ids"].shape[1]), "latency_s": latency}


def parse_letter(reply: str, letters: str) -> str | None:
    """First option letter in the reply, e.g. 'B' from 'B.' or 'The answer is B'."""
    s = reply.strip().upper()
    if s and s[0] in letters:
        return s[0]
    for tok in s.replace(".", " ").replace(")", " ").split():
        if tok in letters:
            return tok
    return None
