from collections import deque

from rc_config import (
    MAX_LOG_LINES,
    MAX_TURNS_TO_KEEP,
    CAMERA_PRIM_PATH,
    LAST_FRAME_PATH,
    SENT_FRAMES_ROOT,
    CAPTURE_FPS,
    FRAME_BUFFER_SECONDS,
)


class State:
    def __init__(self):
        # UI models / widgets
        self.window = None
        self.log_label = None
        self.log_frame = None
        self.input_model = None
        self.attach_latest_model = None
        self.action_label = None
        self.belief_label = None
        self.ltm_label = None

        # Logs / chat
        self.log_lines = deque(maxlen=MAX_LOG_LINES)
        self.chat_history = deque(maxlen=2 * MAX_TURNS_TO_KEEP + 2)

        # Replicator / camera
        self.rp = None
        self.rgb_annot = None
        self.camera_ready = False
        self.active_camera_prim_path = CAMERA_PRIM_PATH
        self.last_rgb = None
        self.last_frame_path = LAST_FRAME_PATH
        self.sent_frames_root = SENT_FRAMES_ROOT
        self.sent_frames_dir = SENT_FRAMES_ROOT

        self.playing = False
        self.run_active = False
        self.run_start_time = 0.0
        self.run_id = 0

        # RAM frame buffer for high-rate capture
        self.frame_buffer = deque(maxlen=max(1, int(CAPTURE_FPS * FRAME_BUFFER_SECONDS)))
        self.capture_task = None
        self.inquiry_task = None

        # Cosmos connection state
        self.cosmos_state = None  # None = unknown, True = connected, False = failed
        self.cosmos_last_detail = ""

        # VLM request guard
        self.vlm_busy = False
        self.vlm_busy_since = 0.0
        self.pause_inquiry_until = 0.0
        self.last_vlm_response = ""
        self.last_infer_time = 0.0
        self.last_auto_infer_ts = 0.0

        # Theory-of-Mind / cognition modules (optional)
        self.belief_manager = None
        self.ghost_visualizer = None
        self.worker_in_q = None
        self.worker_out_q = None
        self.worker_thread = None
        self.worker_poll_task = None

        # Agent memory
        self.short_memory = None
        self.long_memory = None

        # Robot control (optional)
        self.robot_controller = None
        self.robot_update_task = None

        # Hybrid ToM: GT state monitor + ring buffer
        self.state_monitor = None
        self.ring_buffer = None
        self.last_trigger_event = None
        self.pending_user_text = ""  # User input waiting to be processed

        # Keep Kit event subscriptions alive (avoid GC).
        self._kit_subs = []

        # Startup guard: prevent any unintended auto-play until stage is ready.
        self.startup_stage_opened = False
        self.startup_autostop_done = False


STATE = State()
