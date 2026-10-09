# Open questions

Things a dataset or tool lacks compared with what CLAUDE.md assumes. Each entry
says what we did about it for now. Farid decides the real fix.

## 2026-10-07: what is on the SCC

- `/projectnb/proactiveai/data` holds only **CaptainCook4D**: the annotations repo
  plus **11 GoPro videos at 360p**, all from one recipe (activity 4, recordings
  4_2, 4_3, 4_22, 4_24, 4_30, 4_32, 4_35, 4_36, 4_40, 4_43, 4_44). StreamGaze,
  EgoGazeVQA, HD-EPIC and HoloAssist are not downloaded.
- Model cache has Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-3B-Instruct,
  Grounding DINO tiny and SigLIP2. EgoGazeLite weights were added on 2026-10-07.

## 2026-10-07: CaptainCook4D has no eye gaze

- No gaze files in the local copy. The dataset paper (arXiv 2312.14556) lists
  HoloLens2 depth, IMU, head and hand tracking, but not eye gaze.
- What we did: predicted gaze only, as CLAUDE.md already says for this case.
  The "gaze crop, measured" condition cannot run on CaptainCook4D.
- Checked 2026-10-07: the HoloLens2 "spatial" pickle (hl2ss spatial-input format)
  for recording 4_2 has 23,224 records. The eye-gaze valid bit is never set and
  the eye-ray fields are zero. Head pose and hand joints are present. So
  CaptainCook4D has no measured gaze, at least in the released files. The
  official downloader lists no gaze stream either.
- Hand joints exist (HoloLens coordinates, not GoPro). They could become a
  "hands" pointing signal later, on the HoloLens PV video only.

## 2026-10-07: CaptainCook4D has no question-answer items

- It has step segments and error labels, not questions.
- What we did for the smoke test: a 5-way multiple-choice step-recognition task
  built from the step annotations (true step + 4 other steps of the same
  recipe). Prompt is fixed and lives in `src/eval/captaincook4d.py`. This is our
  task, not a benchmark task.
- Needs a decision: which CaptainCook4D tasks to use for the real map (step
  recognition, error detection, both?) and their taxonomy category.

## 2026-10-07: CaptainCook4D has no evidence boxes

- So no oracle crop on this dataset without hand labels.

## 2026-10-07: EgoGazeLite needs the previous true gaze point

- Its full model takes the previous measured gaze as an input. The authors'
  `evaluate.py` feeds the real previous gaze at every step.
- What we did: feed the model its own previous prediction (start at frame
  centre). This is not how its reported accuracy was measured, so its
  predictions on our videos may be worse than the paper suggests.
- Other mismatches: trained on Aria video (square, fisheye, 10 fps gaze);
  our GoPro frames are 16:9 and get squashed to 300x300.
- The model has a built-in centre bias. The smoke test adds a centre-crop
  control so we can tell if "predicted gaze" is just "centre".

## 2026-10-07: the local videos are 360p

- "Full, high resolution" here is only 644x364 pixels (299 tokens per 2 frames).
  A 28%-budget crop already covers 196x336 pixels, more than a quarter of the
  frame. Foveation has little resolution to gain at 360p.
- Needs a decision: download higher-resolution GoPro video for the real map.
  The 4K GoPro files exist in the official downloader (about 4.4 GB per
  recording, so about 50 GB for the 11 local recordings).

## 2026-10-08: other datasets, checked against their own pages

- **StreamGaze** (HF `daeunni/StreamGaze`, CC BY 4.0, open): 8,521 QA items, 10
  task types, 2D gaze in CSVs. No videos: they must come from EGTEA Gaze+,
  EgoExoLearn and HoloAssist separately. No evidence boxes. Not downloaded yet.
- **EgoGazeVQA** (HF `taiyi09/EgoGazeVQA`, gated): 1,757 items, 5 options. Most
  videos come from Ego4D / EgoExo4D, which need a signed licence. Skipped for now.
- **HoloAssist** (CDLA v2, open): gaze 2.45 GB, full-res video 184 GB, mistake
  labels. No evidence boxes. Not downloaded yet.
- **HD-EPIC** (data.bris, readme says CC BY-NC 4.0, open): downloading all 156
  MP4s (124 GB) and the MPS gaze files. 26K 5-way questions.
  - Evidence: no per-question evidence box. But every object movement has a box at
    its pick-up and put-down frame. 7,186 of 10,000 fine-grained action questions
    have such a box inside their time window. We use the union of those boxes as
    the oracle region (an approximation: it marks the moved object, not
    necessarily all the evidence, and it is held fixed over the 1-3 s window).
  - Gaze: Aria MPS yaw/pitch in the glasses' frame, 10 Hz, not pixels. Projected
    into the MP4 with each video's camera calibration, read from the first 200 MB
    of its VRS file (scripts/prep_hdepic_gaze.py). Orientation checked by eye on
    2026-10-09 (6 frames of P01-20240202-110250): the upright projection lands on
    the hands and the object being handled; the raw one lands on background. We
    use upright.
  - Videos are 1408x1408 fisheye. Segments are 1-10 s, so we take a fixed 8
    frames per segment instead of 1 fps (1 fps would give 1-3 frames).
