import rclpy
from rclpy.node import Node
import asyncio
import websockets
import json
import logging
device_id='D058984606732'
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
class WebSocketNode(Node):
    def __init__(self):
        super().__init__('websocket_node')
        self.websocket_url = f'ws://99.suyiiyii.top:45685/ws/device/{device_id}'  # 这里替换为实际的WebSocket地址
        self.reconnect_interval = 5  # 重连间隔时间（秒）
        self.loop = asyncio.get_event_loop()
        # self.create_timer(0.1, self.spin_once)  # 创建定时器来定期执行spin_once
        self.ws_task = self.loop.create_task(self.websocket_manager())  # 启动WebSocket管理任务
        self.loop.call_soon(self.spin_once)

    async def websocket_manager(self):
        while rclpy.ok():  # 只要ROS还在运行，就保持重连机制
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logger.info(f'Connected to WebSocket: {self.websocket_url}')
                    await self.listen(websocket)
            except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK, ConnectionRefusedError,Exception) as e:
                logger.warn(f'Connection lost: {e}. Reconnecting in {self.reconnect_interval} seconds...')
                await asyncio.sleep(self.reconnect_interval)
            # except Exception as e:
            #     await asyncio.sleep(self.reconnect_interval)
            #     logger.warning(e)

    async def listen(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'command':
                    logger.info(f"Received command: {data.get('message')}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")

    def spin_once(self):
        # 允许ROS 2处理其内部队列和事件循环
        rclpy.spin_once(self, timeout_sec=0)
         # 继续调用下一个 spin_once 循环
        self.loop.call_soon(self.spin_once)
def main(args=None):
    rclpy.init(args=args)
    websocket_node = WebSocketNode()

    try:
        websocket_node.loop.run_forever()  # 启动异步事件循环
    except KeyboardInterrupt:
        logger.info('Shutting down WebSocket node...')
    finally:
        websocket_node.ws_task.cancel()  # 取消WebSocket任务
        websocket_node.loop.run_until_complete(websocket_node.ws_task)  # 确保任务已经被取消
        websocket_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()