import json
import queue
import threading
import time

import websocket

from shared.commands import Command

# ── Configuration ─────────────────────────────────────────────────────────────

ROBOT_IP: str = "localhost"
ROSBRIDGE_PORT: int = 9090
ROSBRIDGE_URL: str = f"ws://{ROBOT_IP}:{ROSBRIDGE_PORT}"

TOPIC: str = "/gesture_cmd"
TOPIC_TYPE: str = "std_msgs/String"

# Seconds to wait before a reconnection attempt after a failure.
RECONNECT_DELAY: float = 3.0

# ─────────────────────────────────────────────────────────────────────────────

_ADVERTISE_MSG = json.dumps({
    "op": "advertise",
    "topic": TOPIC,
    "type": TOPIC_TYPE,
})


class GestureSender:
    """Publishes gesture commands to a ROSBridge WebSocket server.

    A background daemon thread owns the WebSocket connection and handles
    reconnection automatically. The main thread calls send() without blocking.
    """

    def __init__(self, url: str = ROSBRIDGE_URL):
        self._url = url
        # maxsize=10: absorbs bursts; stale commands are drained on reconnect.
        self._queue: queue.Queue[str] = queue.Queue(maxsize=10)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Background worker ─────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            try:
                ws = websocket.WebSocket()
                ws.connect(self._url, timeout=5)
                print(f"[GestureSender] Connected to ROSBridge at {self._url}")
                ws.send(_ADVERTISE_MSG)

                while True:
                    msg = self._queue.get()     # blocks until a message is ready
                    ws.send(msg)

            except Exception as exc:
                print(f"[GestureSender] Connection error: {exc}. "
                      f"Retrying in {RECONNECT_DELAY}s…")
                self._drain_queue()
                time.sleep(RECONNECT_DELAY)

    def _drain_queue(self) -> None:
        """Discard queued messages so stale commands don't replay on reconnect."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ── Public API ────────────────────────────────────────────────────────────

    def send(self, command: Command) -> None:
        """Enqueue a command for publication. Non-blocking; drops if queue full."""
        msg = json.dumps({
            "op": "publish",
            "topic": TOPIC,
            "msg": {"data": command.value},
        })
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            pass    # video loop must never block on a network operation

    def close(self) -> None:
        pass    # daemon thread exits with the main process

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
