from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='face_recognition',
            executable='FaceRecognitionPublisher',
            name='face_recognition',
            output='screen'
        ),
        Node(
            package='remote_control',
            executable='remote_control',
            name='remote_control',
            output='screen'
        ),
        Node(
            package='remote_control',
            executable='command_subscriber',
            name='command_subscriber',
            output='screen'
        ),
        # Node(
        #     package='remote_control',
        #     executable='test',
        #     name='test',
        #     output='screen'
        # ),
        Node(
            package='remote_control',
            executable='video_send',
            name='video_send',
            output='screen'
        ),
        # 如果需要添加更多节点，只需继续添加 Node 实例
    ])
