# Implementation Plan (Stage-1) — Scene-Agnostic Triggered VLM + STM/Belief + Ghost USD

## 1) Task Spec (required)
All scene/task particulars are defined in a Task Spec file.

### Task Spec must define
- Sensor sources (camera prim path or attachment rule)
- Entity discovery rules (how to select “trackable entities”)
- Trigger rules (what changes trigger AUTO inquiry)
- Belief targets (which entities require object permanence tracking)
- Rates and sizes:
  - capture rate Hz
  - frames per inquiry K
  - queue coalescing policy

### Entity discovery strategies (choose one or combine)
- Semantics-based: semantic labels / instance ids
- Pattern-based: prim path regex, prim type filters, metadata keys
- Collection-based: USD collections
- GT-enumeration + filter: enumerate rigid bodies/articulations and filter
- Model-assisted (optional): model suggests which entities matter based on a snapshot

---

## 2) Data Structures (generic)
### 2.1 STM JSON
Per entity:
- stable id (prim path or assigned UUID)
- type/kind (prim type, semantics if present)
- pose/velocity (if allowed stage-1 GT)
- visibility flags (vision-derived)
- timestamps + sources

### 2.2 Belief JSON
Per entity:
- hypothesis state
- confidence
- evidence (events, temporal cues, last seen)
- last update timestamp

---

## 3) Timing & Concurrency
### Main thread (sim tick)
- capture frames at rate (timer)
- run trigger detector
- enqueue inquiries
- update ghost prims each tick
- UI updates (chat + log viewer)

### Worker thread/process (inference)
- pops jobs (AUTO priority)
- bundles last K frames
- calls model
- parses response
- updates shared STM/belief

Inference must never block sim tick.

---

## 4) Triggers (stage-1 GT-based, but abstracted)
Trigger module consumes a generic “state vector” per entity:
- pose deltas
- contact events
- visibility changes (optional)
- user-defined signals

Trigger configuration:
- thresholds
- debounce
- coalescing

---

## 5) Ghost USD
- Spawn ghost prim per belief target (from spec or inferred)
- ghost pose updated from belief every tick
- optional confidence visualization later

---

## 6) Manual inquiry
- manual messages enter same queue with lower priority
- manual payload can be last 1 frame or last K frames (spec-driven)
- returns free-form output + optional structured block
