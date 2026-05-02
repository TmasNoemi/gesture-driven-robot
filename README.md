# gesture-driven-robot

A computer vision system that detects hand gestures in real time and translates them into robot commands published over ROS2 via ROSBridge WebSocket.

## Architecture

```
gesture-driven-robot/
├── gesture_module/
│   ├── detector.py       # Hand landmark detection and gesture classification
│   └── sender.py         # ROSBridge WebSocket publisher
├── robot_module/         # Robot simulation and command execution (ROS2)
├── shared/
│   └── commands.py       # Command enum and topic name constants
├── .env                  # Local configuration (not committed)
└── pyproject.toml
```

## How it works

MediaPipe tracks both hands simultaneously from the webcam feed. Each hand is identified by handedness and mapped to a dedicated ROS2 topic:

```
Camera → gesture_module/detector.py
              ├── Left hand  → /gesture_movement  (MOVE / STOP)
              └── Right hand → /gesture_rotation  (MOVE_FORWARD / MOVE_BACKWARD / ROTATE_RIGHT / ROTATE_LEFT)
                                    ↓
                         gesture_module/sender.py
                                    ↓
                         ROSBridge WebSocket (ws://ROBOT_IP:9090)
                                    ↓
                              robot_module (ROS2)
```

## Gesture reference

### Left hand — `/gesture_movement`

| Gesture | Command | Notes |
|---------|---------|-------|
| Open hand (all fingers extended) | `STOP` | Immediate, no smoothing |
| Closed fist | `MOVE` | Confirmed after 5 frames |

### Right hand — `/gesture_rotation`

| Gesture | Command | Notes |
|---------|---------|-------|
| Index up (only index extended, pointing up) | `MOVE_FORWARD` | Confirmed after 5 frames |
| Index down (only index extended, pointing down) | `MOVE_BACKWARD` | Confirmed after 5 frames |
| Fist + clockwise wrist tilt | `ROTATE_RIGHT` | Confirmed after 4 frames |
| Fist + counter-clockwise wrist tilt | `ROTATE_LEFT` | Confirmed after 4 frames |
| Fist held still (no tilt) | — | No command published; robot holds last state |

Direction detection uses 3D landmark coordinates and is independent of palm orientation (front or back of hand facing the camera).

## Configuration

Copy `.env.example` to `.env` and set your robot's IP:

```
ROBOT_IP=192.168.x.x
ROSBRIDGE_PORT=9090
TOPIC=/gesture_cmd
TOPIC_TYPE=std_msgs/String
RECONNECT_DELAY=3.0
```

The sender reconnects automatically if the ROSBridge server is unreachable on startup.

## Setup

```bash
uv sync
uv run python -m gesture_module.detector
```

The MediaPipe hand landmark model (`hand_landmarker.task`) is downloaded automatically on first run.

## ROS2 module

The `robot_module` is a self-contained ROS2 package (`gesture_robot`) that runs inside a Gazebo simulation. It receives gesture commands over WebSocket and drives a simulated differential-drive robot.

### Components

**`rosbridge_websocket`** — listens on port 9090 and bridges WebSocket messages to native ROS2 topics. The gesture module on the PC connects here; no special networking setup is needed beyond knowing the robot's IP.

**`command_bridge.py`** — the core ROS2 node. It subscribes to both gesture topics and translates string commands into `geometry_msgs/Twist` velocity messages on `/cmd_vel`, which the Gazebo differential drive controller reads directly.

The two hands work as an enable/direction pair:

- `/gesture_movement` (left hand) acts as a **movement gate**. `MOVE` enables motion; `STOP` zeroes all velocities immediately regardless of what the right hand is doing.
- `/gesture_rotation` (right hand) sets **direction and speed** while the gate is open. The robot can also rotate in place even when the left hand is `STOP`.

Velocity constants (configurable in `command_bridge.py`):

| Parameter | Value |
|-----------|-------|
| Linear velocity | 0.5 m/s |
| Angular velocity | 0.5 rad/s |

Command-to-Twist mapping:

| Command | `linear.x` | `angular.z` |
|---------|-----------|------------|
| `MOVE` + `MOVE_FORWARD` | +0.5 | 0 |
| `MOVE` + `MOVE_BACKWARD` | −0.5 | 0 |
| `ROTATE_RIGHT` (any movement state) | 0 | −0.5 |
| `ROTATE_LEFT` (any movement state) | 0 | +0.5 |
| `STOP` | 0 | 0 |

**`robot.urdf.xacro`** — describes the robot's physical model:

- Rectangular base (0.4 × 0.3 × 0.1 m, 1 kg)
- Two driven wheels with high friction (radius 5 cm, separation 0.34 m)
- Front caster wheel (frictionless) for stability
- Red front marker for orientation reference

**`simulation.launch.py`** — single launch file that brings up the full stack:

1. Gazebo with `simple_world.sdf` (ground plane + two static walls at y = ±5 m creating a corridor)
2. `robot_state_publisher` (broadcasts the URDF transforms)
3. Robot spawner (places the robot at the origin)
4. `rosbridge_websocket` server
5. `command_bridge` node

### Running the simulation

```bash
# Inside the ROS2 workspace
cd robot_module/ros2_ws
colcon build
source install/setup.bash
ros2 launch gesture_robot simulation.launch.py
```

Once Gazebo is up and ROSBridge is listening on port 9090, start the gesture detector on the PC side:

```bash
uv run python -m gesture_module.detector
```

### Message format

Commands reach ROSBridge as `std_msgs/String` JSON messages:

```json
{ "op": "publish", "topic": "/gesture_movement", "msg": { "data": "MOVE" } }
{ "op": "publish", "topic": "/gesture_rotation", "msg": { "data": "ROTATE_LEFT" } }
```

Odometry is published by the Gazebo differential drive plugin on `/odom` and can be forwarded back to the PC via ROSBridge if needed.
