#!/usr/bin/env python3
"""
command_bridge.py

Subscribes al topic /gesture_command (std_msgs/String) pubblicato da rosbridge
quando Noemi manda un Command enum dal suo script Python su macOS.
Converte il comando in geometry_msgs/Twist e pubblica su /cmd_vel.

Flusso:
  Noemi (macOS) → WebSocket → rosbridge → /gesture_command → qui → /cmd_vel → Gazebo
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


# ── Mappatura Command enum → velocità (linear.x, angular.z) ───────────────
# Modifica questi valori per cambiare velocità/comportamento del robot
COMMAND_MAP = {
    'MOVE_FORWARD':  {'linear_x':  0.5, 'angular_z':  0.0},
    'MOVE_BACKWARD': {'linear_x': -0.5, 'angular_z':  0.0},
    'MOVE_LEFT':     {'linear_x':  0.0, 'angular_z':  0.5},
    'MOVE_RIGHT':    {'linear_x':  0.0, 'angular_z': -0.5},
    'STOP':          {'linear_x':  0.0, 'angular_z':  0.0},
}


class CommandBridge(Node):

    def __init__(self):
        super().__init__('command_bridge')

        # Subscriber: riceve i comandi stringa da rosbridge
        self.subscription = self.create_subscription(
            String,
            '/gesture_command',
            self.command_callback,
            10
        )

        # Publisher: manda i comandi di movimento al robot in Gazebo
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info('CommandBridge avviato. In ascolto su /gesture_command...')

    def command_callback(self, msg: String):
        command = msg.data.strip().upper()

        if command not in COMMAND_MAP:
            self.get_logger().warn(f'Comando sconosciuto ricevuto: "{command}"')
            return

        velocities = COMMAND_MAP[command]

        twist = Twist()
        twist.linear.x = float(velocities['linear_x'])
        twist.angular.z = float(velocities['angular_z'])

        self.publisher.publish(twist)
        self.get_logger().info(f'Comando: {command} → linear.x={twist.linear.x}, angular.z={twist.angular.z}')


def main(args=None):
    rclpy.init(args=args)
    node = CommandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
