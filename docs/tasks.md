# Tasks (Stage-1) — Scene-Agnostic Perception + STM + Belief + Ghost USD

## P0 — Must-Have
1) Task Spec format + loader
- Define schema (YAML/JSON)
- Validate and log configuration at startup

2) Entity discovery module
- Implement at least two discovery paths:
  - (A) semantics/tags/patterns
  - (B) GT enumeration + filter rules
- Output a stable list of tracked entity IDs

3) GT initialization (stage-1)
- Seed STM from GT for tracked entities
- Spawn ghost prims for belief targets

4) Frame capture ring buffer
- Capture at configured Hz
- Store in RAM buffer with timestamps

5) Trigger detector (configurable)
- Detect changes based on configured signals
- Support debounce + thresholding
- Enqueue AUTO jobs

6) Inquiry queue manager
- AUTO > MANUAL priority
- AUTO coalescing
- Worker in-flight management

7) Model call worker (async)
- Bundle last K frames
- Send query
- Parse response robustly (no crash on malformed output)

8) STM/belief updater
- Apply structured updates
- Diff + log

9) Ghost updater
- Update ghost prims smoothly each tick based on belief

---

## P1 — Nice-to-Have
- Confidence visualization
- GT-vs-belief scoring hooks (for evaluation, not for control)
- Manual commands: “dump stm”, “dump belief”, “dump tracked entities”

## P2 — Placeholders
- Sensor-only mode (GT disabled)
- Long-term memory store
- Motion control / planners
