"""Decode and cache the sampled frames for a config, without the VLM.

Decoding 4K HEVC is memory-hungry (GPU jobs were killed at 20-33 GB), so run
this as a CPU batch job before the GPU run; the GPU job then only reads the
cached JPEGs:
  qsub -P proactiveai -pe omp 8 -l mem_per_core=8G -l h_rt=06:00:00 -j y -o <log> \
       <wrapper running: uv run --locked python scripts/prepare_frames.py configs/experiment.yaml 2>
"""

from __future__ import annotations

import sys
import time

import yaml

from eval.captaincook4d import CaptainCook4D
from fovea import views


def main(cfg_path: str, workers: int) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    d = dict(cfg["dataset"], decode_workers=workers)
    full_hw = views.grid_px(tuple(cfg["full_grid"]))
    if d["name"] == "hdepic":
        from eval.hdepic import HDEpic
        ds = HDEpic(d, cfg["sampling"], full_hw, cfg["seed"])
    else:
        ds = CaptainCook4D(d, cfg["sampling"], full_hw, cfg["seed"])
    t0 = time.perf_counter()
    for k in range(0, len(ds.segments), 20):  # chunks, so progress shows in the log
        ds.prepare_frames(ds.segments[k:k + 20])
        print(f"{min(k + 20, len(ds.segments))}/{len(ds.segments)} segments, {time.perf_counter() - t0:.0f}s", flush=True)
    print("FRAMES_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
