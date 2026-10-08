"""Project HD-EPIC (Aria MPS) eye gaze into the MP4 frames.

Run as a batch job, in an isolated env with projectaria-tools (Python 3.12):
  uv run --python 3.12 --isolated --no-project --with projectaria-tools --with opencv-python-headless \
      python scripts/prep_hdepic_gaze.py <hdepic_root> <out_dir> [--check]

For each video:
- the device calibration is read from the first 200 MB of its VRS file (downloaded
  to $TMPDIR and deleted; the calibration sits in the file header),
- gaze (general_eye_gaze.csv, device time) is projected into the RGB camera at the
  MPS vergence depth (1 m when depth is missing),
- device time is mapped to MP4 time with the dataset's *_mp4_to_vrs_time_ns.csv.
Output: <out_dir>/<video>.csv with time_s, x_raw, y_raw, x_up, y_up (pixels of the
1408x1408 frame; "up" = rotated upright as the Aria viewer shows it). Which of the
two matches the MP4 is decided by eye from the --check overlays and recorded in
docs/open_questions.md.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

BASE = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47"


def calibration(vid: str, tmp: Path):
    from projectaria_tools.core import data_provider
    part = vid.split("-")[0]
    head = tmp / f"{vid}.vrs"
    subprocess.run(["curl", "-sL", "--retry", "5", "-r", "0-209715199", "-o", str(head),
                    f"{BASE}/VRS/{part}/{vid}_anonymized.vrs"], check=True)
    try:
        p = data_provider.create_vrs_data_provider(str(head))
        dev = p.get_device_calibration()
        return dev, dev.get_camera_calib("camera-rgb")
    finally:
        head.unlink(missing_ok=True)


def project(root: Path, vid: str, tmp: Path) -> np.ndarray:
    from projectaria_tools.core import mps
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
    part = vid.split("-")[0]
    z = zipfile.ZipFile(root / "SLAM-and-Gaze" / part / "GAZE_HAND" / f"mps_{vid}_vrs.zip")
    name = next(n for n in z.namelist() if n.endswith("general_eye_gaze.csv"))
    gpath = tmp / f"{vid}_gaze.csv"
    gpath.write_bytes(z.read(name))
    gazes = mps.read_eyegaze(str(gpath))
    gpath.unlink()
    dev, rgb = calibration(vid, tmp)
    t_dev, raw, up = [], [], []
    for g in gazes:
        depth = g.depth if g.depth and g.depth > 0 else 1.0
        r = get_gaze_vector_reprojection(g, "camera-rgb", dev, rgb, depth, make_upright=False)
        u = get_gaze_vector_reprojection(g, "camera-rgb", dev, rgb, depth, make_upright=True)
        t_dev.append(g.tracking_timestamp.total_seconds() * 1e9)
        raw.append(r if r is not None else (np.nan, np.nan))
        up.append(u if u is not None else (np.nan, np.nan))
    # device ns -> mp4 seconds
    m = np.loadtxt(root / "Videos" / part / f"{vid}_mp4_to_vrs_time_ns.csv", delimiter=",", skiprows=1)
    t_mp4 = np.interp(np.array(t_dev), m[:, 2], m[:, 0]) / 1e9
    return np.column_stack([t_mp4, np.array(raw), np.array(up)])


def check_overlays(root: Path, vid: str, g: np.ndarray, out: Path) -> None:
    import cv2
    cap = cv2.VideoCapture(str(root / "Videos" / vid.split("-")[0] / f"{vid}.mp4"))
    for t in np.linspace(g[0, 0] + 30, g[-1, 0] - 30, 6):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        i = int(np.argmin(np.abs(g[:, 0] - t)))
        for (x, y), col in (((g[i, 1], g[i, 2]), (0, 0, 255)), ((g[i, 3], g[i, 4]), (0, 255, 0))):
            if np.isfinite(x):
                cv2.circle(f, (int(x), int(y)), 25, col, 6)
        cv2.putText(f, "red=raw green=upright", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(str(out / f"check_{vid}_{t:.0f}.jpg"), cv2.resize(f, (704, 704)))


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    check = "--check" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    vids = sorted(p.stem for p in (root / "Videos").glob("*/*.mp4"))
    if check:
        vids = vids[:1] + [v for v in vids if v.startswith("P05")][:1]
    for vid in vids:
        dst = out / f"{vid}.csv"
        if dst.exists() and not check:
            continue
        try:
            g = project(root, vid, tmp)
        except Exception as e:  # keep going; a missing gaze file just drops that video's gaze conditions
            print(f"{vid} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        with open(dst, "w") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "x_raw", "y_raw", "x_up", "y_up"])
            w.writerows(np.round(g, 3).tolist())
        print(f"{vid} {len(g)} gaze samples, valid {np.isfinite(g[:, 1]).mean():.3f}", flush=True)
        if check:
            check_overlays(root, vid, g, out)


if __name__ == "__main__":
    main()
