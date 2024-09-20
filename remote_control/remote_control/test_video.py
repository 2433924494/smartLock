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
class VideoSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        
        self.subscription = self.create_subscription(
            String,
            'video_stream',
            self.listener_callback,
            10
        )
        logger.info("Test launched!")
        self.bridge = CvBridge()
        self.websocket_clients = set()
    async def run(self):
        # 并行运行 WebSocket 服务器和 ROS 2 spin
        await asyncio.gather(
            self.start_websocket_server(),
            self.spin_ros_node()
        )

    async def spin_ros_node(self):
        # 使用 ROS 2 的 spin 操作
        while rclpy.ok():
            rclpy.spin_once(self)
    def listener_callback(self, msg):
        logger.info('Received video stream frame')
        # frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imwrite(r'E:\ros2_ws\src\test.jpg',frame)
        # _, jpeg_image = cv2.imencode('.jpg', frame)
        # image_base64 = base64.b64encode(jpeg_image).decode('utf-8')
        asyncio.create_task(self.send_to_clients(msg.data))

    async def send_to_clients(self, image_data):
        if self.websocket_clients:
            await asyncio.wait([client.send(image_data) for client in self.websocket_clients])

    async def websocket_handler(self, websocket, path):
        self.websocket_clients.add(websocket)
        try:
            async for message in websocket:
                pass
        finally:
            self.websocket_clients.remove(websocket)

    async def start_websocket_server(self):
        # 启动 WebSocket 服务器并打印确认消息
        start_server = websockets.serve(self.websocket_handler, "0.0.0.0", 8765)
        await start_server
        logger.info("WebSocket server started and listening on ws://0.0.0.0:8765")

        # Keep the server running
        await asyncio.Future()  # 阻塞任务以保持服务器运行

def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriber('video_subscriber_node')

    try:
        # 使用 asyncio.run() 来并行运行 WebSocket 服务器和 ROS 2 spin
        asyncio.run(node.run())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
