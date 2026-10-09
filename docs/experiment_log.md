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

## 2026-10-07: smoke test, first try failed

- The job stopped before the first clip. It could not find Qwen2.5-VL-7B.
- Cause: the SCC `~/.bashrc` sets `TRANSFORMERS_CACHE` to the herbdl folder,
  which overrides our model cache.
- Fix: the script now points all Hugging Face caches at
  `/projectnb/proactiveai/model_cache` itself. Relaunched.

## 2026-10-07: smoke test, second try failed

- The model loaded this time. Then the script crashed on a bug in my code: it
  looked for a `signal` field on the full-frame condition, which has none.
- Fix: use a safe lookup. I then ran the whole script on the login node for 2 clips,
  with a fake model in place of Qwen. Everything else worked: frames, gaze,
  crops, scoring, summary. Token counts matched the plan in every condition
  (4784 full, 448 at the 9% budget, 1344 at the 28% budget).
- Not yet relaunched: two failed runs in a row, so I am checking with Farid first.

## 2026-10-07: smoke test, third try worked (step 1 done)

Run 503f5ee1. 50 step clips, 5-way "which step is this?", Qwen2.5-VL-7B.

Token check: passed. On every one of the 50 clips, all conditions at the same
budget saw exactly the same number of visual tokens (measured from the model
input, not computed). Clips have 4 to 32 frames (14 of 50 hit the 32 cap), so
the token count differs between clips but not between conditions on one clip.

Accuracy (50 clips, so the ranges are wide; 95% bootstrap ranges in brackets):

| Condition | Tokens vs full | Accuracy |
| --- | --- | --- |
| Full frame, 360p | 100% | 72% [60, 84] |
| Shrunk whole frame | 9.4% | 46% [32, 60] |
| Crop at predicted gaze | 9.4% | 48% [34, 60] |
| Crop at centre | 9.4% | 58% [44, 72] |
| Shrunk whole frame | 28.1% | 62% [48, 74] |
| Crop at predicted gaze | 28.1% | 66% [52, 80] |
| Crop at centre | 28.1% | 70% [56, 82] |

What this says, plainly:
- Shrinking the frame costs accuracy: 72% down to 46% at 9% of the tokens.
- At the same token count, cropping is at least as good as shrinking. The centre
  crop is the best of the three at both budgets (+12 points over shrinking at
  9.4%, range -4 to +28; +8 at 28.1%, range 0 to +18). With 50 clips none of
  these gaps is clearly above zero.
- Predicted gaze did no better than shrinking, and worse than a plain centre
  crop. Likely reasons: the gaze model never sees real gaze here (it is fed its
  own guesses), and it was trained on a different camera. So on this data,
  "predicted gaze" is not yet a useful pointing signal.
- Gaze prediction adds about 4.5 s per clip (10 fps over the whole step), more than
  the VLM call itself (0.1 to 1.2 s).

Caveats: one recipe, 11 videos, 360p, our own task, 50 clips.

Check-in with Farid (CLAUDE.md asks for one after step 1): Farid said to keep
going through the whole plan, so I continue and log each step here.

Next:
- The 4K GoPro files for the same 11 videos are downloaded. Step 3 will use
  720p frames made from them, so crops have real detail to gain.
- Step 2 (oracle crop) needs evidence boxes. CaptainCook4D has none. Checking
  the other datasets.

## 2026-10-08: what happened between step 1 and the full map

- More data:
  - CaptainCook4D: downloaded 4K versions of the 11 smoke-test recordings plus
    26 more recordings picked at random across recipes. 3 more were planned but
    their download was cut off by my mistake (I overwrote the file list the
    download job was reading); I dropped them rather than wait.
  - HD-EPIC: downloading all 156 videos (124 GB) plus eye-gaze files. It has
    real eye gaze and object boxes, so it gives us the measured-gaze and
    oracle conditions that CaptainCook4D cannot.
- CaptainCook4D has no eye gaze at all: I opened its HoloLens tracking file and
  the eye-gaze field is empty in all 23,224 records.
- A mistake on my side: I ran a CPU test on the SCC login node and the SCC killed
  it (more than 15 min of CPU). Everything long now goes through batch jobs.
- Task taxonomy written to `docs/task_taxonomy.md` before any step-3 run.
- Pipeline test on one GPU (2 clips, all 21 conditions, real Qwen): token check
  passed everywhere. A full-frame call takes about 12 s (about 19K visual tokens
  for 32 frames at 720p); the 10% and 25% conditions take 0.4 s and 1.4 s.
  The object detector found the named objects in 36% to 100% of frames.

## 2026-10-08: step 3 (full map) on CaptainCook4D launched

Run 078b8b11, one L40S, about 10 hours expected.
- 37 recordings, every step at least 4 s long, two tasks per step:
  "which step is this?" (5 options) and "was there a mistake in this step?" (yes/no).
- Full frame is 720p (1288x728). Budgets: 10% and 25% of the full token count.
- 21 conditions: full; shrunk whole frame; crops at centre, random, predicted
  gaze, named objects, gaze+objects; and each pointing crop again with a small
  shrunk whole frame added (crop + periphery), all at the same token total.
  So this run also covers step 4 (periphery) for this dataset.

## 2026-10-09: full map, first try answered nothing

- Run 078b8b11 waited about 12 hours in the GPU queue, then finished at once
  with 0 clips. Cause: in the config, recording names like 4_2 were not in
  quotes, and YAML reads 4_2 as the number 42. No recording matched.
- Fix: names are quoted, and the script now stops with an error if a recording
  name is unknown or no clips are selected. Relaunched as run 8f545c7b.
- HD-EPIC gaze check: the projected gaze lands on the hands and the object
  being handled in all 6 test frames (picture saved under
  artifacts/checks/hdepic_gaze_projection_check.jpg). Gaze for all videos will be
  projected once the HD-EPIC download finishes (68 of 156 videos so far; the
  first download job hit its 12 h limit, a second one is running).

## 2026-10-09: full map, third try also killed (out of memory)

- Run 025afbdb: even with 4 video readers, pulling frames from the 4K videos
  inside the GPU job used 20 GB and the job was killed. No item scored. Each try
  kept its frames, so 324 of 511 segments are now cached.
- Fix: frames are now pulled in a separate CPU batch job with more memory
  (scripts/prepare_frames.py, 2 readers). The GPU run starts after it and only
  reads the cached pictures.
- Step 2 (oracle pass, HD-EPIC P01-P02) started: 592 questions (581 clips).
