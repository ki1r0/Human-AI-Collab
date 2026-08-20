# Stage B iteration 1 — restore the R1 workspace offset

Classification: **EXPLORATORY**.

## Hypothesis

H6: the rule oracle failed because R1 was paired with the later R1-Lite 0.15 m workspace/table offset. Restoring the R1 and checkpoint-era 0.20 m offset will allow more official relationships, ideally all six, to score.

## Evidence before the run

- The last December 2025 R1 source sets the table, external environment, and agent environment offsets to 0.20 m.
- May 2026 commit `5631142` changed the table and external environment to 0.15 m after the project had switched its default embodiment to R1 Lite.
- The current agent environment still sets 0.20 m.
- The initial oracle run paired R1 with 0.15 m and missed gear 2, gear 4, and carrier/ring relationships.

## Locked change and procedure

Change only the table and external-environment `x_offset` values from 0.15 m to 0.20 m. Preserve seed 2026, R1, contact/rest offsets, randomization, official rule trajectory, score function, timing, cameras, and the 700-step bound. Run the same `run_rule_oracle.py` command.

## Prediction and decision rule

The best score must exceed the prior 3 to support H6. Stage B passes only at score at least 6 with no timeout/error. A result at or below 3 refutes H6 and triggers collision-margin analysis; it does not authorize motion/score changes.
