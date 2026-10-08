"""Build the per-condition frame stacks that are fed to the VLM.

Qwen2.5-VL turns a video into tokens of 2 frames x 28 x 28 pixels. So a frame
stack of T frames at H x W pixels (both multiples of 28) costs exactly
(T / 2) * (H / 28) * (W / 28) visual tokens. Every condition here returns frames
whose size is a whole number of 28-pixel cells, so the token count is fixed by
the grid we choose, not by the resizer inside the model.

A budget is a token grid (rows, cols). Downsampling shrinks the whole frame to
that grid. A crop cuts a window of that grid's size out of the full-resolution
frame, at native pixel scale. Both give the same number of tokens by
construction.
"""

from __future__ import annotations

import cv2
import numpy as np

CELL = 28  # pixels per visual token side in Qwen2.5-VL (14px patch x 2x2 merge)


def grid_px(grid: tuple[int, int]) -> tuple[int, int]:
    """Token grid (rows, cols) -> pixel size (height, width)."""
    return grid[0] * CELL, grid[1] * CELL


def full_frames(frames: np.ndarray) -> np.ndarray:
    """Full-resolution condition: frames are already at the full grid size."""
    return frames


def downsample(frames: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Shrink the whole frame to the budget grid (area interpolation)."""
    h, w = grid_px(grid)
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def crop_at(frames: np.ndarray, centres_xy: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cut a budget-sized window around one point per frame, at native scale.

    The window is shifted (not padded) so it always stays inside the frame.
    Returns the crops and the boxes used, as (x0, y0, x1, y1) per frame.
    """
    ch, cw = grid_px(grid)
    H, W = frames.shape[1:3]
    assert ch <= H and cw <= W, f"crop {ch}x{cw} larger than frame {H}x{W}"
    crops, boxes = [], []
    for f, (cx, cy) in zip(frames, centres_xy):
        x0 = int(round(np.clip(cx - cw / 2, 0, W - cw)))
        y0 = int(round(np.clip(cy - ch / 2, 0, H - ch)))
        crops.append(f[y0:y0 + ch, x0:x0 + cw])
        boxes.append((x0, y0, x0 + cw, y0 + ch))
    return np.stack(crops), np.array(boxes)


def centre_points(n: int, frame_hw: tuple[int, int]) -> np.ndarray:
    """Fixed centre-crop pointing signal."""
    H, W = frame_hw
    return np.tile([[W / 2, H / 2]], (n, 1))


def expected_tokens(n_frames: int, frame_hw: tuple[int, int]) -> int:
    """Visual tokens Qwen2.5-VL should produce for this frame stack."""
    H, W = frame_hw
    assert H % CELL == 0 and W % CELL == 0, (H, W)
    return (n_frames // 2) * (H // CELL) * (W // CELL)


def fit_region(box: tuple[float, float, float, float], grid: tuple[int, int],
               frame_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Smallest window with the grid's aspect ratio that contains `box` and is at
    least the grid's pixel size, shifted to stay inside the frame.

    A point is a zero-size box, so a point gives exactly the native-scale window.
    A large box gives a bigger window that `crop_regions` then shrinks to the grid.
    """
    gh, gw = grid_px(grid)
    H, W = frame_hw
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    scale = max(1.0, bw / gw, bh / gh)
    scale = min(scale, W / gw, H / gh)  # never bigger than the frame
    ww, wh = gw * scale, gh * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0 = float(np.clip(cx - ww / 2, 0, W - ww))
    wy0 = float(np.clip(cy - wh / 2, 0, H - wh))
    return int(round(wx0)), int(round(wy0)), int(round(wx0 + ww)), int(round(wy0 + wh))


def crop_regions(frames: np.ndarray, boxes: np.ndarray, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Crop one region per frame (see fit_region) and resize it to the budget grid."""
    gh, gw = grid_px(grid)
    out, used = [], []
    for f, b in zip(frames, boxes):
        x0, y0, x1, y1 = fit_region(tuple(b), grid, f.shape[:2])
        win = f[y0:y1, x0:x1]
        if win.shape[:2] != (gh, gw):
            win = cv2.resize(win, (gw, gh), interpolation=cv2.INTER_AREA)
        out.append(win)
        used.append((x0, y0, x1, y1))
    return np.stack(out), np.array(used)


def points_to_boxes(points: np.ndarray) -> np.ndarray:
    """(x, y) points -> zero-size boxes."""
    return np.concatenate([points, points], axis=1)


def random_points(n: int, frame_hw: tuple[int, int], grid: tuple[int, int], seed: int) -> np.ndarray:
    """Random-crop control: one uniformly placed window per frame, seeded per item."""
    rng = np.random.default_rng(seed)
    H, W = frame_hw
    gh, gw = grid_px(grid)
    return np.stack([rng.uniform(gw / 2, W - gw / 2, n), rng.uniform(gh / 2, H - gh / 2, n)], axis=1)
