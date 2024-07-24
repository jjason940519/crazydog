from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='focInterface',
            executable='foc_command_sub',
            name='foc_command_sub'
        ),
        Node(
            package='focInterface',
            executable='foc_data_pub',
            name='foc_data_pub'
        ),
    ])