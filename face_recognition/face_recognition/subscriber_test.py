import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Subscriber(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("Hello! I'm %s" % name)
        self.command_subscribe=self.create_subscription(String,'command',self.callback_func,10)
    def callback_func(self,msg):
        speed=0.0
        if msg.data=='backup':
            speed=-0.2
        self.get_logger().info(f'Receive[{msg.data}],Speed:{speed}')

def main(args=None):
    rclpy.init(args=args) # 初始化rclpy
    node = Subscriber("subscriber_test")  # 新建一个节点
    rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy
