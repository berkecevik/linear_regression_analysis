#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        # Launch the analysis launcher node first
        Node(
            package='linear_regression_analysis',
            executable='analysis_launcher',
            name='analysis_launcher_node',
            output='screen'
        ),
        
        # Launch height-weight analysis node with a small delay
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='linear_regression_analysis',
                    executable='height_weight_node',
                    name='height_weight_analysis_node',
                    output='screen'
                )
            ]
        ),
        
        # Launch brain-weight analysis node with a small delay
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='linear_regression_analysis',
                    executable='brain_weight_node',
                    name='brain_weight_analysis_node',
                    output='screen'
                )
            ]
        ),
        
        # Launch boston housing analysis node with a small delay
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='linear_regression_analysis',
                    executable='boston_housing_node',
                    name='boston_housing_analysis_node',
                    output='screen'
                )
            ]
        ),
    ])
