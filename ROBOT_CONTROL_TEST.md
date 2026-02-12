# Robot Control Testing Guide

## Quick Test Commands

### Manual Text Prompts (via UI)
Once the UI is running and VLM is responding, test robot control by typing these in the chat:

```
go to home position
open your gripper
close your gripper
move your end effector to position 0.5, 0.0, 0.6
inspect the scene from above
```

The VLM should parse these and generate action commands like:
```json
{
  "type": "home",
  "args": {}
}
```

---

## Direct Action Testing (Python Console)

If you want to bypass the VLM and test the robot controller directly:

### 1. Access via Script Editor

In Isaac Sim, go to **Window → Script Editor** and run:

```python
import rc_ui
import rc_state

# Test if robot controller is enabled
rc = rc_state.STATE.robot_controller
if rc and rc.enabled:
    print(f"✓ Robot controller ready: {rc.controller_type}")
    print(f"  Prim path: {rc.prim_path}")
else:
    print("✗ Robot controller disabled")
    print("  Check logs for errors")

# Test home action
if rc and rc.enabled:
    rc.set_action({"type": "home", "args": {}})
    print("✓ Sent 'home' action")

# Test gripper
if rc and rc.enabled:
    rc.set_action({"type": "open_gripper", "args": {}})
    print("✓ Sent 'open_gripper' action")
```

### 2. Test End-Effector Movement

```python
import rc_state

rc = rc_state.STATE.robot_controller
if rc and rc.enabled:
    # Move to a safe test position
    rc.set_action({
        "type": "move_ee_pose",
        "args": {
            "pos": [0.5, 0.0, 0.5],  # x, y, z in world frame
            "quat": [1.0, 0.0, 0.0, 0.0]  # w, x, y, z (identity rotation)
        }
    })
    print("✓ Sent move command - watch the robot!")
```

---

## Debugging Robot Controller Issues

### Issue 1: Controller Disabled

**Symptoms:**
```
[WARN] RobotController disabled: ...
```

**Solutions:**
1. Check if Franka exists in scene:
   ```python
   from pxr import Usd
   import omni.usd

   stage = omni.usd.get_context().get_stage()
   # List all prims
   for prim in stage.Traverse():
       print(prim.GetPath())
   ```

2. Set explicit robot path:
   ```bash
   ROBOT_PRIM_PATH=/World/Franka ./isaaclab.sh -p nvidia_tutorial/robot_control.py
   ```

3. Use device="cuda:0" if available (CPU has issues with PhysX views):
   ```bash
   ./isaaclab.sh -p nvidia_tutorial/robot_control.py --device cuda:0
   ```

### Issue 2: No Controller Available

**Symptoms:**
```
[WARN] No task-space controller available
```

**Cause:** Both RMPFlow and DifferentialIK failed to initialize

**Solution:** Robot will fall back to joint-level control (limited functionality)

### Issue 3: Actions Not Executing

**Check:**
1. Is robot initialized?
   ```python
   rc._robot.is_initialized  # Should be True
   ```

2. Is simulation playing?
   - Click PLAY button in toolbar
   - Robot only updates when physics is running

3. Check action status:
   ```python
   status = rc.update()
   print(f"Status: {status.status} - {status.detail}")
   ```

---

## Expected Behavior

### Successful Initialization

Terminal should show:
```
[INFO] RobotController ready (diffik): /World/Franka
```
or
```
[INFO] RobotController ready (rmpflow): /World/Franka
```

### Action Execution

When you send an action, you should see:
```
[INFO] Robot action finished: done (home)
```

The robot should physically move in the simulation.

---

## Controller Types

1. **RMPFlow** (best) - Smooth, collision-aware motion
2. **DifferentialIK** (fallback) - Fast, simple Jacobian-based IK
3. **Joint-level** (last resort) - Direct joint control only

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `'Articulation' object has no attribute '_root_physx_view'` | Fixed! Restart session |
| Robot disabled on CPU | Use `--device cuda:0` |
| Controller not found | Check ROBOT_PRIM_PATH matches your scene |
| Actions timeout | Increase timeout or check joint limits |

---

## VLM Action Format

The VLM should output actions in this format:

```json
{
  "type": "move_ee_pose",
  "args": {
    "pos": [0.5, 0.0, 0.4],
    "quat": [1.0, 0.0, 0.0, 0.0]
  }
}
```

Supported action types:
- `noop` - Do nothing
- `home` - Return to home pose
- `open_gripper` - Open fingers
- `close_gripper` - Close fingers
- `move_ee_pose` - Move end-effector to position
- `inspect` - Move to inspection pose (0.5, 0.0, 0.6)
