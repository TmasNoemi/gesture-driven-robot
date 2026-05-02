from enum import Enum


class Command(Enum):
    # Left hand
    MOVE = "MOVE"
    STOP = "STOP"
    # Right hand
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    ROTATE_LEFT = "ROTATE_LEFT"
    STOP_ROTATION = "STOP_ROTATION"
    # Legacy
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_LEFT = "MOVE_LEFT"


TOPIC_MOVEMENT = "/gesture_movement"
TOPIC_ROTATION = "/gesture_rotation"
