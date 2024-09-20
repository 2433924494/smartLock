import logging
import rclpy
from rclpy.node import Node

# logging.basicConfig(filename=r'./logs.txt',level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
def main(args=None):
    """
    ros2运行该节点的入口函数
    编写ROS2节点的一般步骤
    1. 导入库文件
    2. 初始化客户端库
    3. 新建节点对象
    4. spin循环节点
    5. 关闭客户端库
    """
    rclpy.init(args=args) # 初始化rclpy
    node = Node("node_test")  # 新建一个节点
    node.get_logger().info("大家好,我是node_test.".encode('utf-8'))
    logger.info('大家好')
    rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy