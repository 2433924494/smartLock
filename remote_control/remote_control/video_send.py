import base64
import json
import time
import websockets
import asyncio
from .control import logger
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2


device_id='D058984606732'
class VideoSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.video_frame=None
        self.subscription = self.create_subscription(
            String,
            'video_stream',
            self.listener_callback,
            10
        )
        self.reconnect_interval=5
        logger.info("Test launched!")
        self.websocket=None
        self.websocket_uri = f'ws://99.suyiiyii.top:45685/ws/device/{device_id}'
        
    def listener_callback(self, msg):
        # logger.info(f'Received video stream frame:{len(msg.data)}')
        self.video_frame=msg.data
        
        # frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imwrite(r'E:\ros2_ws\src\test.jpg',frame)
        # _, jpeg_image = cv2.imencode('.jpg', frame)
        # image_base64 = base64.b64encode(jpeg_image).decode('utf-8')
        # asyncio.create_task(self.send_to_clients(msg.data))
        # if self.websocket:
        #     # 使用 asyncio.create_task 来调度异步操作
        #     logger.info("WebSocket is available, scheduling send_frame_to_websocket task")
        #     asyncio.create_task(self.send_frame_to_websocket(self.video_frame))
        # else:
        #     logger.warn("WebSocket is not connected")
    async def send_frame_to_websocket(self, frame):
        while rclpy.ok():
            try:
                message = json.dumps({
                    "type": "video",
                    "message": frame
                })
                await self.websocket.send(message)
                # logger.info(f"Sent video frame to WebSocket:{len(frame)}")
            except Exception as e:
                logger.error(f"Error sending video frame: {str(e)}")
            await asyncio.sleep(1.00/30)  # Adjust for frame rate
    async def send_video_stream(self):
        """Send video stream to WebSocket server"""
        logger.info('Send video Started!')
        while rclpy.ok():
            if self.video_frame:
                try:
                    # logger.info('send video')
                    message = json.dumps({
                        "type": "video",
                        "message": str(self.video_frame)
                    })
                    # await websocket.send(message)
                    await asyncio.wait_for(self.websocket.send(message), timeout=2)
                    # logger.info(f"Sent video frame to WebSocket:{len(self.video_frame)}")
                except asyncio.TimeoutError:
                    logger.error("Timeout while sending video frame")
                    break
                except Exception as e:
                    logger.error(f"Error sending video frame: {str(e)}")
                    break
            else:
                logger.warn('no frame')
                pass
            await asyncio.sleep(1.0/15)  # Adjust for frame rate
    async def connect(self):
        while rclpy.ok():
            try:
                async with websockets.connect(self.websocket_uri) as websocket:
                    logger.info('WebSocket connected!')
                    # async for message in websocket:
                        # await self.handle_message(message)
                    self.websocket=websocket
                    await asyncio.gather(
                        self.send_video_stream(),
                        self.spin_ros_node()
                    )
                    
            except websockets.ConnectionClosed as e:
                logger.error(f'Connection closed: {e.code} - {e.reason}')
            except Exception as e:
                logger.error(f'Error: {str(e)}')

            # 在连接失败或异常时等待一段时间然后重试
            logger.info(f'Retrying connection in {self.reconnect_interval} seconds...')
            await asyncio.sleep(self.reconnect_interval)
    async def spin_ros_node(self):
        # 使用 ROS 2 的 spin 操作
        while rclpy.ok():
            rclpy.spin_once(self)
            await asyncio.sleep(1.00/30)
    def run(self):
        # 并行运行 WebSocket 服务器和 ROS 2 spin
        # await asyncio.gather(
        #     self.connect(),
        #     self.spin_ros_node()
        # )
        asyncio.run(self.connect())
        
def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriber('video_send')

    try:
        # 使用 asyncio.run() 来并行运行 WebSocket 服务器和 ROS 2 spin
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()