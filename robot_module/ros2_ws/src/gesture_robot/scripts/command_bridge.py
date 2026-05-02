#!/usr/bin/env python3
"""
command_bridge.py

Subscribes to two topics published by the gesture sender:
  - /gesture_movement  (left hand):  MOVE | STOP
  - /gesture_rotation  (right hand): MOVE_FORWARD | MOVE_BACKWARD |
                                     ROTATE_LEFT | ROTATE_RIGHT | STOP_ROTATION

Combines the state of both hands and publishes geometry_msgs/Twist on /cmd_vel:
  - STOP (left)  → robot halts regardless of right hand
  - MOVE (left)  → right hand determines direction/rotation
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

TOPIC_MOVEMENT = '/gesture_movement'
TOPIC_ROTATION = '/gesture_rotation'

LINEAR_VELOCITY  = 0.5
ANGULAR_VELOCITY = 0.5


class CommandBridge(Node):

    def __init__(self):
        super().__init__('command_bridge')

        self._moving = False   # left hand: MOVE / STOP
        self._linear_x: float = 0.0   # last linear velocity set by right hand
        self._angular_z: float = 0.0  # last angular velocity set by right hand

        self.create_subscription(String, TOPIC_MOVEMENT, self._movement_callback, 10)
        self.create_subscription(String, TOPIC_ROTATION, self._rotation_callback, 10)

        self._publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info(
            f'CommandBridge avviato. In ascolto su {TOPIC_MOVEMENT} e {TOPIC_ROTATION}...'
        )

    def _movement_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command == 'MOVE':
            self._moving = True
        elif command == 'STOP':
            self._moving = False
        else:
            self.get_logger().warn(f'Comando movimento sconosciuto: "{command}"')
            return
        self._publish()

    def _rotation_callback(self, msg: String):
        command = msg.data.strip().upper()

        if command == 'MOVE_FORWARD':
            self._linear_x =  LINEAR_VELOCITY
            self._angular_z = 0.0
        elif command == 'MOVE_BACKWARD':
            self._linear_x = -LINEAR_VELOCITY
            self._angular_z = 0.0
        elif command == 'ROTATE_LEFT':
            self._angular_z =  ANGULAR_VELOCITY   # _linear_x mantiene il valore precedente
        elif command == 'ROTATE_RIGHT':
            self._angular_z = -ANGULAR_VELOCITY   # _linear_x mantiene il valore precedente
        else:
            self.get_logger().warn(f'Comando direzione sconosciuto: "{command}"')
            return

        self._publish()

    def _publish(self):
        twist = Twist()
        if self._moving:
            twist.linear.x  = self._linear_x
            twist.angular.z = self._angular_z
        else:
            twist.angular.z = self._angular_z  # rotate in place; linear stays 0

        self._publisher.publish(twist)
        self.get_logger().info(
            f'[{"MOVE" if self._moving else "STOP"}] '
            f'linear.x={twist.linear.x}, angular.z={twist.angular.z}'
        )


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
