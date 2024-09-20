#!/usr/bin/env python3
import asyncio
import json
import threading
import rclpy
from rclpy.node import Node
import websockets
device_id='D058984606732'
import logging
from std_msgs.msg import String

log_path=r'E:\ros2_ws\src\logs.log'
# 创建一个日志记录器
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建一个日志格式器
formatter = logging.Formatter('[%(levelname)s] [%(asctime)s]: %(message)s')

# 创建一个终端日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 创建一个文件日志处理器
file_handler = logging.FileHandler(log_path, mode='a',encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
class remoteControl(Node):
    """
    仅远程开门功能
    """
    def __init__(self,name):
        super().__init__(name)
        logger.info("[%s] started!" % name)
        self.websocket_uri = f'ws://99.suyiiyii.top:45685/ws/device/{device_id}'
        # self.subscription = self.create_subscription(
        #     String,
        #     'video_stream',
        #     self.video_callback,
        #     10
        # )
        self.websocket=None
        logger.info("RemoteControl launched!")
        self.video_frame=None
        self.reconnect_interval =5
    # def video_callback(self, msg):
    #     logger.info(f"Received video frame with size: {len(msg.data)}")
    #     self.video_frame = msg.data  # Received Base64 encoded frame
    #     # if self.websocket:
    #     #     try:
    #     #         message = json.dumps({
    #     #             "type": "video",
    #     #             "message": self.video_frame
    #     #         })
    #     #         # await websocket.send(message)
    #     #         await self.websocket.send(message)
                
    #     #         logger.info("Sent video frame to WebSocket")
    #     #     except Exception as e:
    #     #         logger.error(f"Error sending video frame: {str(e)}")
    #     if self.websocket:
    #     # 使用 asyncio.create_task 来调度异步操作
    #         # logger.info("WebSocket is available, scheduling send_frame_to_websocket task")
    #         asyncio.create_task(self.send_frame_to_websocket(self.video_frame))
            
    #     else:
    #         logger.warn("WebSocket is not connected")

    # async def send_frame_to_websocket(self, frame):
    #     try:
    #         message = json.dumps({
    #             "type": "video",
    #             "message": frame
    #         })
    #         await self.websocket.send(message)
    #         # logger.info("Sent video frame to WebSocket")
    #     except Exception as e:
    #         logger.error(f"Error sending video frame: {str(e)}")
    def run(self):
        asyncio.run(self.connect())
        # asyncio.run(self.connect())
        # await asyncio.gather(
        #     self.connect(),
        #     self.spin_ros_node()
        # )
    async def spin_ros_node(self):
        # 使用 ROS 2 的 spin 操作
        logger.info('spin thread start!')
        while rclpy.ok():
            rclpy.spin_once(self)
            await asyncio.sleep(0)  # Give control back to the event loop
    async def start_websocket(self):
        
        try:
            async with websockets.connect(self.websocket_uri) as websocket:
                logger.info('WebSocket connected!')
                self.websocket=websocket
                # self.spin_thread=threading.Thread(target=self.spin_ros_node)
                # self.spin_thread.start()
                
                async for message in websocket:
                    await self.handle_message(message)
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    async def connect(self):
        while rclpy.ok():
            try:
                async with websockets.connect(self.websocket_uri) as websocket:
                    logger.info('WebSocket connected!')
                    # async for message in websocket:
                        # await self.handle_message(message)
                    self.websocket=websocket
                    await asyncio.gather(
                        # self.send_video_stream(websocket),
                        self.receive_commands(websocket),
                        self.spin_ros_node()
                    )
            except websockets.ConnectionClosed as e:
                logger.error(f'Connection closed: {e.code} - {e.reason}')
            except Exception as e:
                logger.error(f'Error: {str(e)}')

            # 在连接失败或异常时等待一段时间然后重试
            logger.info(f'Retrying connection in {self.reconnect_interval} seconds...')
            await asyncio.sleep(self.reconnect_interval)

    async def send_video_stream(self, websocket):
        """Send video stream to WebSocket server"""
        logger.info('Send video Started!')
        while rclpy.ok():
            if self.video_frame:
                try:
                    logger.info('send video')
                    message = json.dumps({
                        "type": "video",
                        "message": self.video_frame
                    })
                    # await websocket.send(message)
                    await asyncio.wait_for(websocket.send(message), timeout=2)
                    logger.info("Sent video frame to WebSocket")
                except asyncio.TimeoutError:
                    logger.error("Timeout while sending video frame")
                except Exception as e:
                    logger.error(f"Error sending video frame: {str(e)}")
            else:
                logger.warn('no frame')
                pass
            await asyncio.sleep(0.1)  # Adjust for frame rate
    async def receive_commands(self, websocket):
        """Receive commands from WebSocket server"""
        logger.info('Receive commands Started!')
        
        async for message in websocket:
            try:
                await self.handle_message(message)
            except Exception as e:
                logger.error(f"Error receiving commands: {str(e)}")
    async def handle_message(self, message):
        """处理来自 WebSocket 的消息"""
        try:
            data = json.loads(message)
            type = data.get('type')
            if type=='command':
                logger.info(f"Received command: {data.get('message')}")
            else:
                logger.info(f"Received message: {message}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
def main(args=None):
    rclpy.init(args=args) # 初始化rclpy
    try:
        node = remoteControl("remote_control")  # 新建一个节点
        node.run()
    finally:
        if node:
            node.destroy_node()
    # rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy
    # try:
    #     # 使用 asyncio.run() 来并行运行 WebSocket 服务器和 ROS 2 spin
    #     asyncio.run(node.run())
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     rclpy.shutdown()
