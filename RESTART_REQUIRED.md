# ⚠️ RESTART REQUIRED

## Why Restart is Needed

All IsaacLab source files have been patched to fix CPU/GPU compatibility, but **Python modules are cached in memory**. Isaac Sim is still using the old bytecode from before the fixes were applied.

## How to Restart

### Option 1: Kill and Restart Isaac Sim
```bash
# In the terminal where Isaac Sim is running:
# Press Ctrl+C to stop Isaac Sim

# Then restart:
cd /workspace/IsaacLab
./isaaclab.sh -p nvidia_tutorial/robot_control.py --usd nvidia_tutorial/simple_room_scene.usd
```

### Option 2: Docker Container Restart (if needed)
```bash
# Exit Isaac Sim first (Ctrl+C)
# Then restart the container
docker restart <container_id>

# SSH back in and launch:
cd /workspace/IsaacLab
./isaaclab.sh -p nvidia_tutorial/robot_control.py --usd nvidia_tutorial/simple_room_scene.usd
```

## What to Expect After Restart

✅ **Animation plays smoothly** - Orange falls with physics
✅ **No numpy/tensor errors** - All 27 physx_view calls now use `_to_torch()`
✅ **Frame capture works** - 3 frames at 5 FPS saved to disk
✅ **VLM gets proper responses** - ~500 tokens available for output
✅ **Terminal logs show**:
```
[INFO] Captured frames: 5/5 (buffer=10)
[INFO] GT trigger: ['Orange_01'] changed (displacement=0.245m)
[VLM] POST ... | images=3 | max_tokens=500
[VLM] Response in 4s | tokens={'prompt_tokens': 1786, 'completion_tokens': 400}
[WORKER] reply='An orange is suspended above a table with two buckets...'
```

## Files That Were Patched (Need Restart to Load)

1. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py`
2. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py`
3. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py`
4. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py`
5. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection.py`
6. `/workspace/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection_data.py`

Total: **27 physx_view getter calls** fixed with `_to_torch()` helper

## Verification

After restart, check for these log lines:
- ✅ NO `TypeError: unsupported operand type(s) for -: 'numpy.ndarray' and 'Tensor'`
- ✅ NO `AttributeError: 'numpy.ndarray' object has no attribute 'clone'`
- ✅ Animation plays (FPS counter updates, orange moves)
- ✅ `[INFO] Captured frames: X/Y (buffer=Z)` appears

If you still see the TypeError after restart, something went wrong with the file edits. Check if the files contain the `_to_torch()` function at the top.
