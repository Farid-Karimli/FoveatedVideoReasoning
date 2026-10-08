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
