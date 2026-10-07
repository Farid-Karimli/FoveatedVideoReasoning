"""Predicted gaze from EgoGazeLite (github.com/m4tteo3000/EgoGazeLite).

The model takes 300x300 frames at 10 fps. Its full form also takes the previous
*measured* gaze point. CaptainCook4D has no eye tracking, so we feed back the
model's own previous prediction instead (autoregressive rollout, starting at the
frame centre). This is a deviation from how the authors evaluate it; see
docs/open_questions.md.

Frames are squashed from 16:9 to 300x300 as in the original pipeline (which
resizes without padding), and predicted points are mapped back to frame pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from egogazelite.models.gaze_lite import GazeLite

SIZE = 300
GAZE_FPS = 10.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_gaze_model(checkpoint: str, device: str) -> GazeLite:
    """Load an EgoGazeLite checkpoint from the Hugging Face hub (cached in HF_HOME)."""
    path = hf_hub_download("m4tteo3000/EgoGazeLite", checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = GazeLite(
        pretrained_backbone=False,
        use_skip_connections=True,
        use_center_bias=True,
        center_bias_sigma=cfg.get("stage1", {}).get("center_bias_sigma", 0.4),
        fusion_mode=cfg.get("stage3", {}).get("fusion_mode", "residual"),
    )
    # Training used torch.compile / DDP, which prefixes keys; strip them (same as upstream evaluate.py).
    strip = lambda k: k.replace("_orig_mod.", "").replace("module.", "")
    own = model.state_dict()
    state = {strip(k): v for k, v in ckpt["model_state_dict"].items()}
    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    if len(state) != len(own):
        raise RuntimeError(f"EgoGazeLite: loaded {len(state)}/{len(own)} tensors")
    model.load_state_dict(state)
    return model.to(device).eval()


def _prep(frame_rgb: np.ndarray, device: str) -> torch.Tensor:
    f = cv2.resize(frame_rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return ((t - MEAN) / STD).to(device)


@torch.no_grad()
def predict_gaze(model: GazeLite, video_path: str, t_start: float, query_times: np.ndarray,
                 out_hw: tuple[int, int], device: str) -> np.ndarray:
    """Run the gaze model at 10 fps from t_start to the last query time.

    Returns one (x, y) point per query time, in pixels of a frame of size out_hw.
    """
    from egogazelite.models.gaze_lite import FIXATION_IDT_WINDOW

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000.0)
    t_end = float(query_times.max())
    prev, gaze_prev, hidden, history = None, torch.tensor([[SIZE / 2, SIZE / 2]], device=device), None, []
    times, points = [], []
    next_t = t_start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 < next_t:
            continue  # skip to the next 10 fps tick
        next_t += 1.0 / GAZE_FPS
        cur = _prep(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), device)
        if prev is not None:
            history.append(gaze_prev)
            history = history[-FIXATION_IDT_WINDOW:]
            _, coords, hidden, _ = model(cur, prev, gaze_prev, hidden=hidden,
                                         gaze_history=history, fixation_method="idt")
            gaze_prev = coords.float()
        times.append(t)
        points.append(gaze_prev[0].cpu().numpy().copy())
        prev = cur
        if t >= t_end:
            break
    cap.release()
    if not times:
        raise RuntimeError(f"no frames decoded from {video_path} at {t_start}s")
    times, points = np.array(times), np.array(points)
    idx = np.abs(times[None, :] - query_times[:, None]).argmin(axis=1)
    H, W = out_hw
    return points[idx] * np.array([W / SIZE, H / SIZE])
