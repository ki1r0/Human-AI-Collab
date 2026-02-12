import os

# -----------------------------
# SETTINGS
# -----------------------------
# Mem0 key used by this demo.
# Intentionally fixed here to avoid stale shell env values from old sessions.
# Set to empty string to use SQLite fallback (recommended if you don't have a valid mem0 key)
MEM0_API_KEY = "m0-ipsqlnTToPUmVXdn4DNLzpaZH2RbmCKwwsdOVOws"

CAMERA_PRIM_PATH = "/Franka/head_camera"  # Copy Prim Path
TABLE_PRIM_PATH = os.getenv("TABLE_PRIM_PATH", "/World/Table")
ROBOT_PRIM_PATH = os.getenv("ROBOT_PRIM_PATH", "").strip()  # optional: existing Franka prim path in stage
ROBOT_PRIM_EXPR = os.getenv("ROBOT_PRIM_EXPR", "").strip()  # optional regex expression to find robot prim(s)

# Franka control
# Enabled by default. Auto-discovers Franka/Panda robot in the scene.
# Disable by setting ENABLE_FRANKA_CONTROL=0 if your scene has non-Franka articulations.
ENABLE_FRANKA_CONTROL = os.getenv("ENABLE_FRANKA_CONTROL", "1").strip() not in ("0", "false", "False")
FRANKA_HOME_JOINT_POS = {
    # Default Panda home pose (matches many examples). Override in code if your robot differs.
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}

RESOLUTION = (640, 480)
IMAGE_FORMAT = "JPEG"
IMAGE_MIME = "image/jpeg"
IMAGE_QUALITY = 90
SAVE_EACH_SENT_FRAME = True  # Re-enabled for debugging (async save via asyncio.to_thread)

# Saved frames
LAST_FRAME_PATH = "/workspace/IsaacLab/nvidia_tutorial/tmp/isaac_cam_latest.jpg"
SENT_FRAMES_ROOT = "/workspace/IsaacLab/nvidia_tutorial/tmp/sent_frames"
MEMORY_ROOT = "/workspace/IsaacLab/nvidia_tutorial/tmp/memory"
LTM_SQLITE_PATH = os.path.join(MEMORY_ROOT, "ltm.sqlite3")

# Cosmos / OpenAI-compatible endpoint (same env vars as h1_LLM_demo.py)
COSMOS_URL = os.getenv("COSMOS_URL", "").strip()  # optional full URL override
COSMOS_HOST = os.getenv("COSMOS_HOST_IP", "172.17.0.1")
COSMOS_PORT = int(os.getenv("COSMOS_PORT", "8000"))
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "/models/Cosmos-Reason2-8B")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY", "EMPTY")
TIMEOUT_SEC = float(os.getenv("COSMOS_TIMEOUT", "20.0"))
FORCE_JSON_RESPONSE = os.getenv("FORCE_JSON_RESPONSE", "1").strip() not in ("0", "false", "False")

MAX_TURNS_TO_KEEP = 12
MAX_LOG_LINES = 400

# If you truly want ZERO system guidance, keep this None.
SYSTEM_MSG = None
# Example if you want slight help:
# SYSTEM_MSG = "You are a helpful embodied assistant. Use the camera image when provided."

# Prompt for periodic inquiry
STREAM_DEFAULT_PROMPT = "Describe what you see."

# Capture/inquiry defaults
CAPTURE_FPS = 5.0  # 0.2s interval
INQUIRY_INTERVAL_SEC = 1.0  # 1 Hz inquiry
INQUIRY_FRAME_COUNT = 3  # Reduced from 5 to 3 to fit within 2048 token context limit
FRAME_BUFFER_SECONDS = 2.0

# Agent scheduling
MIN_INFER_INTERVAL_SEC = float(os.getenv("MIN_INFER_INTERVAL_SEC", "0.8"))

# ---------------------------------------------------------------------------
# Hybrid Theory-of-Mind: GT State Monitor
# ---------------------------------------------------------------------------
GT_POSITION_THRESHOLD = float(os.getenv("GT_POSITION_THRESHOLD", "0.02"))  # meters
GT_ORIENTATION_THRESHOLD = float(os.getenv("GT_ORIENTATION_THRESHOLD", "0.05"))  # radians
GT_COOLDOWN_SEC = float(os.getenv("GT_COOLDOWN_SEC", "1.0"))  # min time between GT triggers
RING_BUFFER_CAPACITY = int(os.getenv("RING_BUFFER_CAPACITY", "5"))

# Tracked object prim paths (comma-separated, or empty for auto-discover).
GT_TRACKED_PRIMS = [p.strip() for p in os.getenv("GT_TRACKED_PRIMS", "").split(",") if p.strip()]

# ---------------------------------------------------------------------------
# Robot Control Tuning (AI-CPS industrial-grade values)
# ---------------------------------------------------------------------------
# These override the default FRANKA_PANDA_HIGH_PD_CFG values to reduce jitter.
# Source: IsaacLab Robotiq config + FrankaCabinet benchmark.
ROBOT_SHOULDER_STIFFNESS = float(os.getenv("ROBOT_SHOULDER_STIFFNESS", "400.0"))
ROBOT_SHOULDER_DAMPING = float(os.getenv("ROBOT_SHOULDER_DAMPING", "80.0"))
ROBOT_FOREARM_STIFFNESS = float(os.getenv("ROBOT_FOREARM_STIFFNESS", "400.0"))
ROBOT_FOREARM_DAMPING = float(os.getenv("ROBOT_FOREARM_DAMPING", "80.0"))
ROBOT_HAND_STIFFNESS = float(os.getenv("ROBOT_HAND_STIFFNESS", "2000.0"))
ROBOT_HAND_DAMPING = float(os.getenv("ROBOT_HAND_DAMPING", "100.0"))
# PhysX solver iterations (higher = more stable, more expensive).
SOLVER_POSITION_ITERATIONS = int(os.getenv("SOLVER_POSITION_ITERATIONS", "12"))
SOLVER_VELOCITY_ITERATIONS = int(os.getenv("SOLVER_VELOCITY_ITERATIONS", "1"))
# Use DifferentialIK as fallback when RMPFlow/Lula is unavailable.
PREFER_DIFFIK = os.getenv("PREFER_DIFFIK", "0").strip() not in ("0", "false", "False")
