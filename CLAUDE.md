# Foveated Video: when does looking at less help?

Owner: Farid Karimli (BU, PhD). Status: v0, nothing run yet.

## What this project is

A measurement study. Video-language models spend most of their tokens on pixels that do not matter for the task. "Foveation" means giving high resolution to a small region and little or nothing to the rest. Several papers do this with one pointing signal (gaze, the question, learned redundancy) on one task. Nobody has mapped, across tasks, **when foveation helps and which pointing signal works for which task**.

This project builds that map first. A method paper may follow from what the map shows. Do not build a new architecture in this repo until the map exists.

## Hypothesis (fixed before any experiment)

The accuracy kept under foveation depends on one thing: **whether the task's evidence lies inside the region the pointing signal selects.** Task type, model and signal matter only through that.

Predictions that follow:

1. Tasks with small, local evidence (object state, hand-object interaction, small text, attributes) keep most of their accuracy at a fraction of the tokens.
2. Global tasks (layout, counting, spatial relations) lose accuracy under any crop, and recover only with a coarse periphery.
3. Tasks whose evidence is off-attention (a peripheral event, something missing, a wrong object elsewhere) fail under gaze crops but not under object crops, or the reverse.
4. No single pointing signal wins everywhere. Gaze and task-named objects fail on different items.

If the data contradicts these, report that. A clean negative result is a valid outcome.

## Research questions, in priority order

1. **Map.** For each task category, how much accuracy does each foveation condition keep at a matched token budget?
2. **Explain.** Does evidence coverage (is the evidence inside the fovea?) predict the kept accuracy across tasks and signals?
3. **Periphery.** How much does a coarse view of the whole frame recover, and for which tasks?
4. **Signals.** Where do gaze, task-named objects and the question disagree, and which is right when they do?
5. **Mistakes (secondary).** On procedural video, is disagreement between gaze and the objects the current step expects a cue for a mistake?

## Conditions

All conditions feed the same frozen VLM. Compare at **matched visual-token budgets** (start with about 10% and 25% of full).

| Condition | Input |
| --- | --- |
| Full, high resolution | Whole frame, native resolution. Upper bound |
| Full, downsampled | Whole frame shrunk to the token budget. The baseline to beat |
| Centre crop | Fixed crop at frame centre |
| Random crop | Control |
| Gaze crop, measured | Crop around eye-tracker gaze |
| Gaze crop, predicted | Crop around predicted gaze (EgoGazeLite or similar) |
| Object crop | Crop around objects named by the question or the current step, found by an open-vocabulary detector |
| Gaze + object | Union or best box of the two |
| Oracle crop | Crop around annotated evidence, where the dataset has it |
| Each crop + coarse periphery | The crop plus a low-resolution whole frame, total tokens matched |

The fair comparison is always against "Full, downsampled" at the same token count, not against "Full, high resolution".

## Setup

- **Model:** Qwen2.5-VL-7B first (its dynamic resolution makes token budgets easy to control). Add a second model family only after the first full pass.
- **Detector:** Grounding DINO or YOLO-World. Tracking with SAM 2 if boxes flicker.
- **Gaze predictor:** the EgoGazeLite code, `github.com/m4tteo3000/EgoGazeLite`.
- **Sampling:** 1 fps to start. Keep frame count fixed across conditions so only the spatial allocation changes.
- **Compute:** BU Shared Computing Cluster. No fine-tuning in this phase.
- **Storage:** data and models live on the SCC, not in git. Data: `/projectnb/proactiveai/data`. Model cache: `/projectnb/proactiveai/model_cache`.

## Datasets

Verify every claim in this table against the dataset's own documentation before using it. Several are from memory or secondary sources.

| Dataset | Why | To verify |
| --- | --- | --- |
| StreamGaze (arXiv 2512.01707) | Streaming egocentric tasks with gaze, ten task types | Licence, which source videos, how gaze is stored |
| EgoGazeVQA (arXiv 2509.07447) | Intent questions with gaze | Same |
| HD-EPIC | Kitchen video with gaze and fine-grained questions | Gaze format, question categories |
| HoloAssist | Unscripted mistakes, eye gaze | Mistake labels, gaze availability per clip |
| CaptainCook4D | Scripted cooking errors, step annotations; Farid's main dataset | **Whether gaze is included at all.** If not, use predicted gaze only |

Each benchmark item needs a **task category** label. Use the benchmark's own categories, then map them to: local-evidence, global, off-attention, temporal-only. Record the mapping in `docs/task_taxonomy.md` and do not change it after seeing results.

## Metrics

- Accuracy per task category and condition, with confidence intervals (bootstrap over items).
- Kept accuracy: condition accuracy divided by full high-resolution accuracy.
- Visual tokens and wall-clock latency per item, measured, not inferred from pixel counts.
- **Evidence coverage:** share of items whose evidence region overlaps the fovea. Where no evidence annotation exists, label a random sample of 100 items per category by hand and say so.
- The headline plot: kept accuracy against evidence coverage, one point per (task category, signal).

## Experiment order

1. **Smoke test.** One dataset, 50 items, three conditions (full high resolution, full downsampled, gaze crop). Confirm token counts are really matched.
2. **Oracle pass.** Oracle crop against full downsampled on every dataset that has evidence annotation. If the oracle crop does not beat downsampling anywhere, stop and report: the idea has no ceiling.
3. **Full map.** All conditions, all datasets, two token budgets.
4. **Periphery ablation.** Crop alone against crop plus coarse periphery.
5. **Coverage analysis.** Fit kept accuracy on evidence coverage. Report the fit and the outliers.
6. **Disagreement analysis.** Items where gaze and object crops differ; which one holds the evidence.
7. **Second model.** Repeat step 3 on one more model family.
8. **Mistake cue (secondary).** Gaze against step-expected objects on HoloAssist and CaptainCook4D.

Stop and check in with Farid after steps 1, 2 and 3.

## Prior work

Position against these. Entries marked (title only) have not been read in full; read them before citing.

- **Gaze crops:** GazeLLM (arXiv 2504.00221), EgoGazeLite (2608.15614), GazeVLM (2509.16476).
- **Gaze in streaming models:** StreamGaze (2512.01707), GazeQwen (2603.25841). GazeQwen gains 26 to 34 points on some StreamGaze tasks and under 2 on others; that split motivates this study.
- **Gaze plus objects:** gaze and Set-of-Mark for interaction anticipation (2604.03667). Overlays, not token reduction.
- **Question-driven foveation:** Foveated Reasoning (2604.21079), images only.
- **Learned selection:** AutoGaze (2603.12254).
- **Object tokens:** VideoOrion (2411.16156); detector-empowered video LLM (2512.06673).
- **Query-aware pruning:** EgoPrune (2507.15428), Q-Frame (2506.22139); Vista-LLM, CROP, Object-Centric Vision Token Pruning (title only).
- **Joint frame and pixel allocation:** "Rethinking Long-Video Efficiency" (2610.04318) (title only; read first, it may overlap).
- **Robotics:** Look, Focus, Act (2507.15833), a foveated transformer with a coarse periphery; Oat-VLA (2509.23655), object plus gripper tokens.
- **Gaze and mistakes:** Gazing Into Missteps (2406.08379). Baseline for step 8.
- **Shortcut use in egocentric models:** arXiv 2607.08514.

Before writing any novelty claim, rerun a prior-art search. This list is from early October 2026.

## Rules for the agent

- Write the hypothesis and task taxonomy to disk before running step 3. Do not edit them afterwards; add dated notes instead.
- Never compare conditions at different token counts without saying so in the same sentence.
- Log every run: config, commit hash, dataset version, token counts, seed. One row per run in `results/runs.csv`.
- Report what was measured. If a number comes from fewer than 100 items, say how many.
- Do not tune prompts per condition. One fixed prompt per benchmark, taken from the benchmark where it provides one.
- Do not claim a result generalises beyond the model and datasets tested.
- If a dataset lacks something this file assumes (gaze, evidence boxes, categories), stop and note it in `docs/open_questions.md` instead of working around it silently.
- Keep data out of git. Commit as Farid-Karimli <karimli.farid@icloud.com>.

## Layout

```
configs/      one YAML per condition and dataset
src/fovea/    crop and periphery builders, one per pointing signal
src/eval/     benchmark loaders, scoring, token and latency counters
scripts/      entry points for each experiment step
results/      runs.csv, per-run JSON, plots
docs/         task_taxonomy.md, open_questions.md, notes
```

## Related projects

- **Swarm Perception:** its watchers already localise step-relevant objects. They can serve as the object pointing signal here.
- **Deadline Change Detection:** if foveated inputs are used for mistake scores, emit the shared score-stream format (`video_id, t_start, t_end, score, source`).
