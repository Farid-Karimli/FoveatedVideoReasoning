# Task taxonomy

Written 2026-10-08, before any step-3 (full map) run. Do not edit existing
mappings after results are seen. New datasets get their own section, added and
committed before their first run. Corrections go in "Dated notes" at the end.

The four categories (from CLAUDE.md):

- **local-evidence**: the answer is visible in a small region, usually where
  the hands are (object state, hand-object interaction, small text, attributes).
- **global**: the answer needs the whole scene (layout, counting, spatial relations).
- **off-attention**: the evidence is somewhere the wearer is probably not
  looking (peripheral event, something missing, a wrong object elsewhere).
- **temporal-only**: the answer depends on order or timing across the video,
  not on what is visible in any one region.

## CaptainCook4D (our tasks; the dataset has no question benchmark)

| Our task | Unit | Category | Why |
| --- | --- | --- | --- |
| Step recognition (5-way: which step is this?) | one annotated step segment | local-evidence | The step is defined by the hands and the object being handled |
| Error detection (2-way: did the person make a mistake in this step?) | one annotated step segment | per error type, below | |

Error types use CaptainCook4D's own labels (`error_annotations.csv`). A step can
carry several types; an error item counts toward every type it carries.

| CaptainCook4D error type | Category | Why |
| --- | --- | --- |
| Technique Error | local-evidence | How the hands do the action (e.g. "peeled improperly") |
| Measurement Error | local-evidence | Amount of an ingredient being handled |
| Preparation Error | local-evidence | State of the ingredient being handled (e.g. "do not peel the onion") |
| Temperature Error | off-attention | Shown on a dial or display, often away from the hands |
| Order Error | temporal-only | Depends on which steps came before; not visible inside one step clip |
| Timing Error | temporal-only | Duration (e.g. "microwave for 1 min instead of 4") |
| Missing Step | temporal-only | Depends on the whole recording |
| Other | unassigned | Too mixed to place |

Error-free step items have no category. They enter the task-level score
(balanced accuracy) but not the per-category scores.

## HD-EPIC (added 2026-10-08, before any HD-EPIC run)

We use only the question types whose input is one short segment of one video
(1 to 10 s). The long-video types (3D perception, object motion, recipe,
ingredient, nutrition) need whole recordings and are out of scope for now.
HD-EPIC's own grouping is the file prefix (fine-grained, gaze).

| HD-EPIC question type | HD-EPIC group | Category | Why |
| --- | --- | --- | --- |
| fine_grained_action_recognition | Fine-grained action | local-evidence | Which action, decided by hands and the object handled |
| fine_grained_how_recognition | Fine-grained action | local-evidence | How the hands do it |
| fine_grained_why_recognition | Fine-grained action | temporal-only | Purpose depends on what comes before and after the clip |
| gaze_gaze_estimation | Gaze | global | Answers are places described relative to other fixtures ("cupboard right of the microwave") |
| gaze_interaction_anticipation | Gaze | local-evidence | The next object is usually where the wearer already looks |

## Other datasets

Not yet mapped. Each gets a section here before its first run.

## Dated notes

(none yet)
