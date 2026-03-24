# Debug Guidelines — Regression Avoidance (Updated)

## 0) Prime directive
**Minimize changes to the current working simple-room demo.**  
Prefer **adding** new modules/files and **integrating via small hooks** over editing existing logic.

> You *can* touch working code when necessary — but treat it as a **last resort** and keep edits surgical, feature-flagged, and easy to revert.

---

## 1) Change strategy: add > edit > refactor
### ✅ Preferred (lowest regression risk)
- Add new functionality in **new files/modules**
- Integrate by adding a **thin hook** in the demo (one call site)
- Guard new behavior behind **flags** (OFF by default)

### ⚠️ Acceptable when necessary
- Small edits to working scripts to:
  - expose a callback/hook point
  - pass through handles (world, scene, sensors, robot prim)
  - register update functions on tick / play events
- Edits must be:
  - minimal diff
  - no renames/moves unless unavoidable
  - reversible in one commit

### 🚫 Avoid unless you have no alternative
- Refactors “for cleanliness”
- Renaming prim paths / reorganizing USD hierarchy
- Changing the main simulation loop structure
- Rewriting the working model connector if it already works

---

## 2) “Thin hook” integration pattern
The working demo should only do:
1) load scene as before
2) (optional) call `pipeline.install(context)` once
3) (optional) call `pipeline.tick(dt)` each frame

All heavy logic lives in new modules.

---

## 3) Feature flags (OFF by default)
Required toggles:
- `ENABLE_GT_INIT=0/1`
- `ENABLE_CAPTURE=0/1`
- `ENABLE_TRIGGER=0/1`
- `ENABLE_GHOSTS=0/1`
- `ENABLE_AUTOINFER=0/1`
- `ENABLE_UI_CONSOLE=0/1`

If anything breaks, flipping flags to `0` must restore baseline behavior.

---

## 4) Regression gates (must pass before any merge)
### Gate A — Baseline (flags OFF)
- Launch the demo with all new flags OFF
- Expected:
  - scene loads
  - play runs
  - no new errors
  - existing behavior unchanged

### Gate B — Incremental enable
Enable one feature at a time:
1. GT init
2. capture
3. trigger
4. ghosts
5. auto-infer

Run each for a fixed duration (e.g., 60s) without crashes, freezes, or runaway memory.

### Gate C — Scene-agnostic smoke test
Run the pipeline on a different simple scene using the same code, only changing the task spec/config.
Expected: discovery + STM init + trigger + inquiry + ghosts all still work.

---

## 5) Logging requirements
Each module logs:
- init/start/stop + effective config
- capture FPS, inquiry rate, queue sizes
- model request/response latency + parse success/fail
- STM/belief update summaries
Logs must appear in the UI console (or be mirrored there).

---

## 6) Common regression causes (avoid)
- Hardcoding object names/paths (“orange”, “basket”) instead of discovery rules
- Blocking HTTP/model calls on the main thread
- Writing frames to disk synchronously
- Changing USD prim paths the demo already depends on
- Increasing rates without bounds or backpressure

---

## 7) If regression happens
1) Disable all flags → confirm baseline works
2) Enable features one-by-one → isolate
3) Fix inside the new module first (avoid touching demo)
4) Only edit the demo if there’s no other way, and keep it minimal
