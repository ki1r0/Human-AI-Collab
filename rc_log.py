import time

import carb

from rc_state import STATE


def render_log():
    if STATE.log_label is None:
        return
    STATE.log_label.text = "\n".join(STATE.log_lines)

    # Try to auto-scroll to bottom (best-effort; depends on Kit build)
    try:
        if STATE.log_frame is not None:
            STATE.log_frame.scroll_y = 1e9
    except Exception:
        pass


def log_line(prefix, msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {prefix} {msg}"
    STATE.log_lines.append(line)
    render_log()


def log_info(msg):
    log_line("INFO ", str(msg))
    carb.log_info(str(msg))


def log_warn(msg):
    log_line("WARN ", str(msg))
    carb.log_warn(str(msg))


def log_error(msg):
    log_line("ERROR", str(msg))
    carb.log_error(str(msg))
