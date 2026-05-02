import math
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

from shared.commands import Command
from gesture_module.sender import GestureSender

_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Static gesture thresholds ─────────────────────────────────────────────────

# Min (TIP–MCP 3D distance) / palm_size to consider a finger extended.
FINGER_EXTENSION_RATIO: float = 0.30

# Min magnitude of the 3D MCP(5)→TIP(8) vector to accept a pointing gesture.
POINTING_MIN_MAGNITUDE: float = 0.08

# Min value of the dominant nx/ny component of the normalized 3D direction
# vector. Rejects gestures where the finger points mostly into/out of screen.
DOMINANT_THRESHOLD: float = 0.40

# Consecutive identical frames required to confirm a static gesture.
SMOOTHING_FRAMES: int = 5

# ── Fist / rotation thresholds ────────────────────────────────────────────────

# Max (TIP–MCP 3D distance) / palm_size for a finger to count as "closed".
# Slightly above FINGER_EXTENSION_RATIO so partially-curled fingers qualify.
FIST_RATIO: float = 0.40

# Min signed angle (radians) from the captured neutral to emit a rotation.
# ~20° dead zone prevents drift from triggering commands.
ROTATION_TILT_THRESHOLD: float = 0.35

# Consecutive identical frames required to confirm a rotation command.
# Kept short (4) so the robot reacts quickly to a held tilt.
ROTATION_SMOOTHING_FRAMES: int = 4

# ─────────────────────────────────────────────────────────────────────────────

_ROTATION_COMMANDS = {Command.ROTATE_RIGHT, Command.ROTATE_LEFT}


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        print(f"Downloading hand landmark model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


def _dist3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _extended(lm, tip: int, mcp: int, palm_size: float) -> bool:
    return _dist3(lm[tip], lm[mcp]) / palm_size > FINGER_EXTENSION_RATIO


# ── Fist detection ────────────────────────────────────────────────────────────

def _is_fist(lm) -> bool:
    """Return True when all five fingers are closed (closed fist).

    Uses TIP–MCP distance normalised by palm_size. Works regardless of
    hand orientation or wrist angle — no anatomy-specific tuning needed.
    """
    palm_size = _dist3(lm[0], lm[9])
    if palm_size < 1e-6:
        return False
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]   # (tip, mcp) per finger
    fingers_closed = all(
        _dist3(lm[tip], lm[mcp]) / palm_size < FIST_RATIO
        for tip, mcp in pairs
    )
    thumb_tucked = _dist3(lm[4], lm[5]) / palm_size < FIST_RATIO
    return fingers_closed and thumb_tucked


# ── Wrist-tilt rotation detection ────────────────────────────────────────────

def _knuckle_axis_angle(lm) -> float:
    """Angle (radians) of the index-MCP → pinky-MCP axis in screen coords."""
    return math.atan2(lm[17].y - lm[5].y, lm[17].x - lm[5].x)


def _classify_tilt(current: float, neutral: float) -> Command | None:
    """Compare current knuckle-axis angle against the captured neutral.

    Uses atan2(sin, cos) to compute the shortest signed difference so the
    ±π discontinuity is handled correctly.

    In y-down screen coordinates:
        diff > 0  →  clockwise tilt from neutral  →  ROTATE_RIGHT
        diff < 0  →  counter-clockwise tilt       →  ROTATE_LEFT
    """
    diff = math.atan2(math.sin(current - neutral), math.cos(current - neutral))
    if diff > ROTATION_TILT_THRESHOLD:
        return Command.ROTATE_RIGHT
    if diff < -ROTATION_TILT_THRESHOLD:
        return Command.ROTATE_LEFT
    return None


# ── Static gesture classification ────────────────────────────────────────────

def _classify(lm) -> Command | None:
    """Classify STOP and pointing gestures from 3D landmarks."""
    palm_size = _dist3(lm[0], lm[9])
    if palm_size < 1e-6:
        return None

    thumb  = _extended(lm, 4,  2,  palm_size)
    index  = _extended(lm, 8,  5,  palm_size)
    middle = _extended(lm, 12, 9,  palm_size)
    ring   = _extended(lm, 16, 13, palm_size)
    pinky  = _extended(lm, 20, 17, palm_size)

    if thumb and index and middle and ring and pinky:
        return Command.STOP

    if not (index and not middle and not ring and not pinky):
        return None

    dx = lm[8].x - lm[5].x
    dy = lm[8].y - lm[5].y
    dz = lm[8].z - lm[5].z

    magnitude = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if magnitude < POINTING_MIN_MAGNITUDE:
        return None

    nx = dx / magnitude
    ny = dy / magnitude

    if abs(ny) >= abs(nx):
        if abs(ny) < DOMINANT_THRESHOLD:
            return None
        return Command.MOVE_FORWARD if ny < 0 else Command.MOVE_BACKWARD
    else:
        if abs(nx) < DOMINANT_THRESHOLD:
            return None
        return Command.MOVE_RIGHT if nx > 0 else Command.MOVE_LEFT


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_landmarks(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 128, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


# ── Detector class ────────────────────────────────────────────────────────────

class GestureDetector:
    """Detects static and dynamic hand gestures from BGR frames."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_ms = int(time.time() * 1000)

        _maxlen = max(SMOOTHING_FRAMES, ROTATION_SMOOTHING_FRAMES)
        self._history: deque[Command | None] = deque(maxlen=_maxlen)

        # Knuckle-axis angle (radians) captured on the first fist frame.
        # None when no fist is currently held.
        self._neutral_angle: float | None = None

    def detect(self, bgr_frame) -> Command | None:
        """Process one BGR frame and return the confirmed Command, or None.

        Static gestures require SMOOTHING_FRAMES identical frames.
        Rotation gestures require ROTATION_SMOOTHING_FRAMES identical frames
        on top of the wrist-roll angle accumulation.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._neutral_angle = None
            self._history.append(None)
            return None

        landmarks = result.hand_landmarks[0]
        _draw_landmarks(bgr_frame, landmarks)

        if _is_fist(landmarks):
            current = _knuckle_axis_angle(landmarks)
            if self._neutral_angle is None:     # first fist frame: capture neutral
                self._neutral_angle = current
                raw = None
            else:
                raw = _classify_tilt(current, self._neutral_angle)
        else:
            self._neutral_angle = None          # fist released: reset neutral
            raw = _classify(landmarks)

        self._history.append(raw)

        n = ROTATION_SMOOTHING_FRAMES if raw in _ROTATION_COMMANDS else SMOOTHING_FRAMES
        recent = list(self._history)[-n:]
        if len(recent) == n and len(set(recent)) == 1:
            return recent[0]
        return None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run(camera_index: int = 0) -> None:
    """Open the webcam, detect gestures and publish them via ROSBridge.

    Connects to ROSBridge on startup (non-blocking — reconnects automatically
    in the background if the server is not yet reachable). Press Q to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    with GestureDetector() as detector, GestureSender() as sender:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            command = detector.detect(frame)

            if command is not None:
                sender.send(command)

            label = command.value if command else "—"
            cv2.putText(
                frame, label, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow("Gesture Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
