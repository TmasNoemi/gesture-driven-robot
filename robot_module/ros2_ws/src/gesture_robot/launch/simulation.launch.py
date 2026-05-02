import os
import xacro
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg = get_package_share_directory('gesture_robot')

    # ── Arguments ──────────────────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg, 'worlds', 'simple_world.sdf'),
        description='Path to the Gazebo world file'
    )

    # ── Gazebo ─────────────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ]),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    # ── Robot State Publisher (loads URDF) ─────────────────────────────────
    urdf_path = os.path.join(pkg, 'urdf', 'robot.urdf.xacro')
    robot_description = xacro.process_file(urdf_path).toxml()
    robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    name='robot_state_publisher',
    parameters=[{'robot_description': robot_description}]
)

    # ── Spawn robot in Gazebo ──────────────────────────────────────────────
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'gesture_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5',  # alzato da 0.1 a 0.5
        ],
        output='screen'
    )

    # ── rosbridge WebSocket server (porta 9090) ────────────────────────────
    # Noemi si connette qui da macOS con il suo script Python
    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{
            'port': 9090,
            'address': '',          # accetta connessioni da qualsiasi IP
            'ssl': False,
            'authenticate': False,
        }],
        output='screen'
    )

    # ── command_bridge: legge i comandi e pubblica su /cmd_vel ────────────
    command_bridge = Node(
        package='gesture_robot',
        executable='command_bridge.py',
        name='command_bridge',
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        rosbridge,
        command_bridge,
    ])
