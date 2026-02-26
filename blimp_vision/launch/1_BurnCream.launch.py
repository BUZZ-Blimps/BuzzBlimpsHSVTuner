#!/usr/bin/env python3
"""
Preset launch file for the SillyAh robot.
This launch file wraps the base node.launch.py and sets preset parameters,
including the namespace ('SillyAh') and a fixed camera_number ('12').
"""

from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the base launch file (node.launch.py) in your package.
    base_launch_file = PathJoinSubstitution([
        get_package_share_directory('blimp_vision'),
        'launch',
        'node.launch.py'
    ])

    # Create a group that pushes our preset namespace.
    group = GroupAction([
        PushRosNamespace('BurnCream'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([base_launch_file]),
            launch_arguments={
                'camera_number': '4',
                'device_path': '/dev/video0',
                'calibration_path': PathJoinSubstitution([
                    get_package_share_directory('blimp_vision'),
                    'calibration'
                ]),
                'ball_model_file': PathJoinSubstitution([
                    get_package_share_directory('blimp_vision'),
                    'models/balloon/balloon_v11n_rknn_model'
                ]),
                'goal_model_file': PathJoinSubstitution([
                    get_package_share_directory('blimp_vision'),
                    'models/goal/goal_v11n_rknn_model'
                ]),
                'verbose_mode': 'false',
                'save_frames': 'false',
                'save_location': 'frames/',
                'input_video_path': ""
            }.items()
        )
    ])

    return LaunchDescription([group])
