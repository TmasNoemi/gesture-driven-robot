# gesture-driven-robot

A computer vision system that detects hand gestures in real time and translates them into commands for a simulated robot.

## Architecture

```
gesture-driven-robot/
├── gesture_module/       # Hand detection and gesture recognition
│   └── __init__.py
├── robot_module/         # Robot simulation and command execution
│   └── __init__.py
├── shared/               # Shared definitions used by both modules
│   ├── __init__.py
│   └── commands.py       # Command enum (STOP, MOVE_*, ...)
└── README.md
```

### Modules

**`gesture_module`**
Captures video input, detects hand landmarks, and maps gestures to `Command` values from `shared.commands`.

**`robot_module`**
Receives `Command` values and drives the simulated robot accordingly (movement, stopping).

**`shared`**
Contains contracts shared across modules — currently the `Command` enum — to avoid circular imports and keep both modules decoupled.

### Data flow

```
Camera → gesture_module → Command → robot_module → Robot action
```

### Commands

| Command         | Description              |
|-----------------|--------------------------|
| `STOP`          | Stop all movement        |
| `MOVE_FORWARD`  | Move the robot forward   |
| `MOVE_BACKWARD` | Move the robot backward  |
| `MOVE_LEFT`     | Move the robot left      |
| `MOVE_RIGHT`    | Move the robot right     |
