"""
Launch the full task-execution pipeline:
  - face_detection node
  - detect_rings node
  - speech_servicer node
  - visualiser node
  - task_executor node  (the main orchestrator)

Usage
-----
  ros2 launch rins_robot task_executor.launch.py
  ros2 launch rins_robot task_executor.launch.py use_sim_time:=true

Patrol waypoints
----------------
Pass a flat list of alternating x/y values in the map frame:
  ros2 launch rins_robot task_executor.launch.py \
      patrol_waypoints:="[1.5,0.0, 1.5,1.5, 0.0,1.5, -1.5,1.5, -1.5,0.0, -1.5,-1.5, 0.0,-1.5, 1.5,-1.5]"

If omitted, the node uses its built-in default rectangle (see task_executor.py).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ARGUMENTS = [
    DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        choices=['true', 'false'],
        description='Use simulation (Gazebo) clock'
    ),
    DeclareLaunchArgument(
        'approach_dist', default_value='0.8',
        description='Stop distance (m) from a detected face/ring'
    ),
    DeclareLaunchArgument(
        'patrol_waypoints', default_value='[0.0]',
        description='Flat [x0,y0, x1,y1, …] patrol waypoints in map frame'
    ),
]


def generate_launch_description():
    use_sim_time     = LaunchConfiguration('use_sim_time')
    approach_dist    = LaunchConfiguration('approach_dist')
    patrol_waypoints = LaunchConfiguration('patrol_waypoints')

    face_detection = Node(
        package='rins_robot',
        executable='face_detection.py',
        name='detect_faces',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    ring_detection = Node(
        package='rins_robot',
        executable='detect_rings.py',
        name='detect_rings',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    speech_servicer = Node(
        package='rins_robot',
        executable='speech_servicer.py',
        name='speech_servicer',
        output='screen',
    )

    visualiser = Node(
        package='rins_robot',
        executable='visualiser.py',
        name='visualizeMarkers',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    task_executor = Node(
        package='rins_robot',
        executable='task_executor.py',
        name='task_executor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'approach_dist': approach_dist,
            'patrol_waypoints': patrol_waypoints,
        }],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(face_detection)
    ld.add_action(ring_detection)
    ld.add_action(speech_servicer)
    ld.add_action(visualiser)
    ld.add_action(task_executor)
    return ld
