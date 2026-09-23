# Scene authoring contract

The ten files in `recipes/` are the pre-Isaac scene contract for the five matched
tasks. They are not a claim that the binary source USDs have already been physically
calibrated. A scene builder must consume one recipe and create a USD layer that:

1. references the listed repository part assets for appearance;
2. creates the visible mutation feature from `shared_mutation_parameters`;
3. creates collision geometry from the same parameter record, using compound pieces
   only where the solver requires decomposition;
4. records the task/variant only in evaluator metadata, never in model-facing text; and
5. runs the paired oracle calibration before the variant is admitted to a model run.

This separation is intentional: a visually plausible but physically mismatched USD is
more damaging than a missing scene, because it can create a false causal result.
