import time

MAX_IDLE = 30 * 60  # 30 minutes

def cleanup_kbs(kbs: dict):
    now = time.time()
    for kb_id in list(kbs.keys()):
        if now - kbs[kb_id]["last_used"] > MAX_IDLE:
            del kbs[kb_id]
