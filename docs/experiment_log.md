# Experiment log

Plain notes on what we did, what we saw, and what is next. Newest entry last.

## 2026-10-07: setup

What is on the SCC:
- Only CaptainCook4D: 11 GoPro videos at 360p, all from one recipe, plus annotations.
- No eye gaze in this dataset. No questions. No evidence boxes.
- Qwen2.5-VL-7B is in the model cache.

What I built:
- A step-recognition test from CaptainCook4D: show one step clip, pick the right
  step from 5 options. 50 clips, picked at random (seed 0).
- Frames: 1 per second, at most 32 per clip.
- Conditions, all on the same frozen Qwen2.5-VL-7B:
  - full frame (299 tokens per 2 frames)
  - full frame shrunk to 28 tokens (9%) and 84 tokens (28%)
  - crop around predicted gaze, 28 and 84 tokens
  - crop at the centre, 28 and 84 tokens (extra control)
- Gaze comes from EgoGazeLite, fed its own previous guess because there is no real gaze.
- The script counts the tokens the model really sees and stops if they differ
  from the plan. A CPU check on the login node gave exact matches
  (4784 / 448 / 1344 tokens for a 32-frame clip).

Gaps are listed in `docs/open_questions.md`.

Next: run the smoke test on one L40S.
