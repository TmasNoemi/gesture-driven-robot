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

_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Configurable thresholds ───────────────────────────────────────────────────

# Min (TIP–MCP 3D distance) / palm_size to consider a finger extended.
FINGER_EXTENSION_RATIO: float = 0.30

# Min magnitude of the 3D MCP(5)→TIP(8) vector to accept a pointing gesture.
POINTING_MIN_MAGNITUDE: float = 0.08

# Min value of the dominant component (nx or ny) of the *normalized* 3D
# direction vector. Rejects gestures where the finger points mostly into/out
# of the screen (large z) with only a small x/y component.
DOMINANT_THRESHOLD: float = 0.40

# Number of consecutive frames that must show the same gesture before it is
# confirmed and returned. Eliminates single-frame flickers.
SMOOTHING_FRAMES: int = 5

# ─────────────────────────────────────────────────────────────────────────────


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        print(f"Downloading hand landmark model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


def _dist3(a, b) -> float:
    """3D Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _extended(lm, tip: int, mcp: int, palm_size: float) -> bool:
    return _dist3(lm[tip], lm[mcp]) / palm_size > FINGER_EXTENSION_RATIO


def _classify(lm) -> Command | None:
    """Classify a hand gesture using 3D landmark coordinates."""
    palm_size = _dist3(lm[0], lm[9])   # wrist(0) → middle-finger MCP(9)
    if palm_size < 1e-6:
        return None

    thumb  = _extended(lm, 4,  2,  palm_size)
    index  = _extended(lm, 8,  5,  palm_size)
    middle = _extended(lm, 12, 9,  palm_size)
    ring   = _extended(lm, 16, 13, palm_size)
    pinky  = _extended(lm, 20, 17, palm_size)

    # ── STOP: all five fingertips far from their MCP joint ────────────────────
    if thumb and index and middle and ring and pinky:
        return Command.STOP

    # ── Pointing: only index finger extended ─────────────────────────────────
    if not (index and not middle and not ring and not pinky):
        return None

    # 3D direction vector: MCP(5) → TIP(8)
    dx = lm[8].x - lm[5].x
    dy = lm[8].y - lm[5].y
    dz = lm[8].z - lm[5].z

    magnitude = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if magnitude < POINTING_MIN_MAGNITUDE:
        return None

    # Normalize to unit vector and classify by dominant screen component.
    # The frame is already flipped (cv2.flip), so screen-x directly maps to
    # the intended direction for any hand and any palm orientation — no further
    # axis correction is needed.
    nx = dx / magnitude
    ny = dy / magnitude

    if abs(ny) >= abs(nx):                      # vertical dominates
        if abs(ny) < DOMINANT_THRESHOLD:
            return None
        return Command.MOVE_FORWARD if ny < 0 else Command.MOVE_BACKWARD
    else:                                       # horizontal dominates
        if abs(nx) < DOMINANT_THRESHOLD:
            return None
        return Command.MOVE_RIGHT if nx > 0 else Command.MOVE_LEFT


def _draw_landmarks(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 128, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)


class GestureDetector:
    """Detects hand gestures from BGR frames using MediaPipe HandLandmarker."""

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
        self._history: deque[Command | None] = deque(maxlen=SMOOTHING_FRAMES)

    def detect(self, bgr_frame) -> Command | None:
        """
        Process a single BGR frame and return the confirmed Command, or None.

        A command is confirmed only when the last SMOOTHING_FRAMES frames all
        agree on the same gesture. This suppresses single-frame flickers.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._history.append(None)
            return None

        landmarks = result.hand_landmarks[0]
        _draw_landmarks(bgr_frame, landmarks)

        raw = _classify(landmarks)
        self._history.append(raw)

        # Confirm only when all recent frames agree
        if len(self._history) == SMOOTHING_FRAMES and len(set(self._history)) == 1:
            return self._history[0]
        return None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run(camera_index: int = 0) -> None:
    """Open the webcam, show confirmed gesture label on screen. Press Q to quit."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    with GestureDetector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            command = detector.detect(frame)

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
