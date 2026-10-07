# Hypothesis (fixed before any experiment)

Copied from CLAUDE.md on 2026-10-07, before any run. Do not edit; add dated notes at the end.


The accuracy kept under foveation depends on one thing: **whether the task's evidence lies inside the region the pointing signal selects.** Task type, model and signal matter only through that.

Predictions that follow:

1. Tasks with small, local evidence (object state, hand-object interaction, small text, attributes) keep most of their accuracy at a fraction of the tokens.
2. Global tasks (layout, counting, spatial relations) lose accuracy under any crop, and recover only with a coarse periphery.
3. Tasks whose evidence is off-attention (a peripheral event, something missing, a wrong object elsewhere) fail under gaze crops but not under object crops, or the reverse.
4. No single pointing signal wins everywhere. Gaze and task-named objects fail on different items.

If the data contradicts these, report that. A clean negative result is a valid outcome.

## Dated notes

(none yet)
