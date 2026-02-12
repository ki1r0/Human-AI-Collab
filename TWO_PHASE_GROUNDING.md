# Two-Phase Grounding Architecture

## Overview

The perception pipeline now operates in **two distinct phases**:

### Phase 1: Initialization (Pre-Simulation)
- **When**: Before pressing PLAY, while the scene is static
- **Trigger**: Click the **"Initiate Cosmos"** button in the UI
- **What happens**:
  1. Captures a single static frame of the current scene
  2. Sends a special **grounding prompt** to the VLM
  3. VLM identifies all visible objects and assigns **semantic names**
  4. Creates initial belief state with named objects (e.g., "orange", "grey_basket", "wooden_table")

### Phase 2: Runtime (During Simulation)
- **When**: After pressing PLAY, while physics is running
- **Trigger**: Automatic (GT state changes, periodic inquiry, manual prompts)
- **What happens**:
  1. Captures a sequence of frames showing temporal changes
  2. Sends an **update prompt** that includes the current belief state
  3. VLM compares frames, detects changes, and **updates** the existing belief
  4. Preserves object names from Phase 1, only updates status/properties

---

## Usage Workflow

### Step 1: Setup the Scene
1. Open Isaac Sim with your USD scene (e.g., `simple_room_scene.usd`)
2. Run the robot_control.py script:
   ```bash
   ./isaaclab.sh -p nvidia_tutorial/robot_control.py --usd nvidia_tutorial/simple_room_scene.usd
   ```
3. Wait for the UI window ("HeadCam Terminal Chat") to appear
4. Click **"Init Camera"** to initialize the camera

### Step 2: Ground the VLM (Initialization)
1. **IMPORTANT**: Do NOT press PLAY yet - keep the scene static
2. Click **"Initiate Cosmos"**
3. Watch the terminal for:
   ```
   [INFO] ═══ Initiating Cosmos Grounding ═══
   [INFO] ✓ Static frame captured for grounding
   [VLM] POST http://172.17.0.1:8000/v1/chat/completions | images=1 | ...
   [VLM] Response in 3.2s | tokens={...} | content_len=512
   [INFO] ═══ Grounding Complete (3.2s) ═══
   [AI  ] I observe an orange fruit on a wooden table...
   ```
4. Check the **Belief (compact)** section in the UI - you should see named objects:
   ```
   orange: visible conf=0.85 | grey_basket: visible conf=0.80 | wooden_table: visible conf=0.90
   ```

### Step 3: Run the Simulation
1. Now press **"PLAY"** to start physics
2. The VLM will automatically receive **update prompts** that preserve the object names from initialization
3. Watch for temporal changes:
   ```
   [INFO] GT trigger: ['orange'] changed (displacement=0.245m)
   [AI  ] The orange rolled from the table into the grey_basket
   [INFO] Belief update: {"objects": {"orange": {"belief_status": "contained", "inferred_container": "grey_basket", ...}}}
   ```

---

## Prompt Modes

### Initialization Mode (Phase 1)
```
INITIALIZATION MODE: Analyze this static scene carefully.

Your task is to identify ALL visible objects and create an initial belief state.

CRITICAL REQUIREMENTS:
1. Use SEMANTIC object names (e.g., 'orange', 'grey_basket', 'white_basket', 'table', 'cup')
2. NEVER use generic IDs like 'object_1', 'object_2', etc.
3. If you see multiple similar objects (e.g., two baskets), distinguish them by color or position
4. For each object, provide:
   - belief_status: 'visible' (since this is a static frame)
   - confidence: your confidence level (0.0-1.0)
   - visible: true
   - temporal_change: 'initial_state'
```

### Runtime Mode (Phase 2)
```
UPDATE MODE: You already have a belief state with established object names.
Your task: Detect temporal changes and UPDATE the existing state.

CURRENT BELIEF STATE (preserve these object names!):
{"objects": {"orange": {...}, "grey_basket": {...}, ...}}

CRITICAL RULES:
1. PRESERVE object names from the current belief state
2. Update belief_status based on what you observe: visible → moving → contained → occluded
3. Update temporal_change to describe what changed in this sequence
4. If an object disappeared, mark visible=false and belief_status='occluded'
5. If a new object appears, add it with a semantic name (not object_N)
```

---

## Benefits of Two-Phase Grounding

### Before (Single-Phase):
- ❌ VLM had to re-identify objects in every frame sequence
- ❌ Generic object names ("object_1", "object_2") were common
- ❌ Object identity could drift across updates
- ❌ No stable reference for tracking

### After (Two-Phase):
- ✅ VLM identifies objects once in a clean, static frame
- ✅ Semantic names established early ("orange", "grey_basket")
- ✅ Object identity preserved across all updates
- ✅ Runtime prompts focus on **what changed**, not **what exists**
- ✅ More accurate containment/occlusion tracking

---

## Debugging Tips

### Check VLM Responses
All VLM calls now log to terminal with `[VLM]` prefix:
```bash
[VLM] POST http://172.17.0.1:8000/v1/chat/completions | images=1 | payload=234.5KB
[VLM] Response in 3.2s | tokens={'prompt_tokens': 1234, 'completion_tokens': 256} | content_len=512
[VLM] Raw content: {"reply": "I see an orange...", "action": {...}, ...}
```

### Check Worker Processing
Worker outputs log with `[WORKER]` prefix:
```bash
[WORKER] reply='I see an orange on the table near a grey basket.'
[WORKER] raw.reply='I see an orange...' | raw.meta={'num_images': 1, 'temporal_summary': 'initial_state'}
```

### Common Issues

**Issue**: VLM still uses "object_1" after grounding
- **Cause**: Grounding didn't complete before PLAY was pressed
- **Fix**: Wait for "═══ Grounding Complete ═══" log before pressing PLAY

**Issue**: Grounding button does nothing
- **Cause**: Camera not initialized
- **Fix**: Click "Init Camera" first

**Issue**: Runtime updates lose object names
- **Cause**: Belief state was reset (STOP button was clicked)
- **Fix**: Click "Initiate Cosmos" again after STOP/reset

---

## Files Modified

1. **`rc_ui.py`**:
   - Added `on_initiate_cosmos()` callback
   - Added "Initiate Cosmos" button to UI
   - Modified worker poll loop to detect grounding completion
   - Added `STATE.grounding_complete` and `STATE.grounding_in_progress` flags

2. **`rc_reason2_agent.py`**:
   - Added init mode detection in `build_reason2_messages()`
   - Split prompt into 3 branches: init mode, runtime mode (with belief), fallback
   - Init mode emphasizes object identification with semantic names
   - Runtime mode emphasizes change detection and name preservation

3. **`rc_vlm.py`**:
   - Added comprehensive `[VLM-SIMPLE]` debug logging
   - Logs HTTP status, response time, token usage, raw content

---

## Next Steps

### Future Enhancements
1. **Visual feedback**: Add a UI label showing "Grounding: ✅ Complete" or "❌ Not initialized"
2. **Belief persistence**: Save initialized belief state to disk so it survives restarts
3. **Re-grounding**: Add a "Re-ground" button to update object names if scene changes significantly
4. **Multi-camera**: Extend grounding to multiple viewpoints for better 3D understanding
5. **Active perception**: VLM requests specific camera movements to resolve ambiguities

### Testing Checklist
- [ ] Grounding identifies all expected objects with semantic names
- [ ] Runtime updates preserve object names from grounding
- [ ] Containment events correctly reference container names (e.g., "orange" → "grey_basket")
- [ ] Occlusion tracking maintains object identity even when not visible
- [ ] Manual prompts during runtime respect the initialized belief state
