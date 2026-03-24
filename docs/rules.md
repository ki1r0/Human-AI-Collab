# Rules for the Isaac Sim Agent Pipeline (Scene-Agnostic, Current Stage)

## 0) Scope (what we are building now)
A reusable Isaac Sim pipeline for:
- Perception from sensor frames (camera frames first)
- Short-term memory (STM) as structured state
- Belief for partially observable states (object permanence / occlusion)
- Belief visualization using Ghost USD prims
- Mixed interaction: automatic (change-triggered) + manual chat

This pipeline MUST work in a new scene by changing only configuration (Task Spec), without editing core code.

---

## 1) Configuration-Driven Only (No Hardcoding)
Core code MUST NOT hardcode:
- object names/classes (e.g., “orange”, “basket”)
- prim paths
- fixed semantic labels
- scene dimensions
- task-specific triggers

All task-specific content MUST live in a Task Spec file (YAML/JSON) or in runtime discovery.

---

## 2) GT Access Policy (Allowed Mode)
GT access is allowed at this stage.

### Allowed GT Uses
- Initialize environment understanding
- Verify model observations (name/pose sanity checks)
- Provide localization for tracked entities

### Required GT Hygiene
Every fact must carry a source tag:
- `source="GT"` for simulator facts
- `source="vision"` for perception from frames
- `source="belief"` for inferred hidden states

GT must not silently overwrite belief; belief updates must remain explicit.

---

## 3) Timing & Update Loop Rules
### Capture cadence
- Sensor frames captured at a configurable rate (default target: 5 Hz)
- Frames stored in a RAM ring buffer (deque), not written to disk by default

### Auto inquiry trigger
- Auto inquiry is triggered by “state change” events detected by a configurable trigger module.
- Each auto inquiry sends the latest `K` frames (default K=5) from the ring buffer.

### Manual inquiry
- Manual inquiries can be submitted anytime (free-form chat)
- Manual inquiries are queued behind auto inquiries

---

## 4) Inquiry Queue Rules (Auto > Manual)
- Use a single queue with priorities:
  - AUTO > MANUAL
- AUTO events must be coalesced:
  - keep only the most recent pending AUTO job while an inference is in-flight
- Manual messages can accumulate but must be rate-limited and non-blocking.

Inference MUST NOT block the simulation loop.

---

## 5) Memory & Belief Definitions (General)
### STM (Short-term memory)
A structured representation of “what the agent currently knows” about tracked entities.
- Mostly seeded from GT at init (stage-1)
- Updated by perception and verification
- Must be serializable, diffable, and logged

### Belief
A structured representation of inferred hidden state, e.g., for occluded or missing objects.
Belief must include:
- `hypothesis` and `confidence`
- `evidence` (temporal reasoning / change cues)
- `valid_until` or recency info

---

## 6) Ghost USD Rules (Belief Visualization)
- A ghost prim is created for each belief-tracked entity (as configured or inferred)
- Ghost prim pose reflects belief, not necessarily GT
- Ghost updates happen in the sim tick (for smooth visualization)
- Confidence can optionally control ghost appearance (later)

---

## 7) Model I/O Contract (Pragmatic)
Manual chat can be free-form, but auto pipeline needs structured outputs.

### Auto inquiry output (preferred)
- natural language summary of temporal change
- machine-readable JSON block containing:
  - `observations`
  - `events` (temporal changes detected)
  - `stm_update`
  - `belief_update`

Parsing failures must not crash the sim; degrade gracefully.

---

## 8) Logging Rules (must-have)
Log:
- capture timestamps and buffer stats
- triggers and trigger reasons
- queue status (AUTO/MANUAL)
- inference latency and errors
- STM/belief diffs
- ghost updates

---

## 9) Portability Requirement
Given a new USD scene:
- pipeline must run by updating Task Spec only
- if no semantic labels exist, pipeline must still operate using prim metadata patterns or GT discovery

