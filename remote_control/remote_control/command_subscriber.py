import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import base64
import websockets
import asyncio
from .control import logger
def opendoor():
    pass
class CommandSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        
        self.subscription = self.create_subscription(
            String,
            'faceAuthentication',
            self.listener_callback,
            10
        )
        logger.info("OpenDoor Listen launched!")
        self.pause=False
    def listener_callback(self,msg):
        if self.pause:
            return
        logger.info(f'Opendoor Listener recv:{msg.data}')
        if msg.data=='known':
            opendoor()
            self.pause = True
            self.create_timer(10, self.end_pause)  # Ends pause after 5 seconds
    def end_pause(self):
        self.pause=False
def main(args=None):
    rclpy.init(args=args)
    node = CommandSubscriber('command_subscriber_node')
    rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy