import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Publisher(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info("Hello! I'm %s" % name)
        self.publisher=self.create_publisher(String,'command',10)
        self.timer=self.create_timer(0.5,self.time_callback)
    def time_callback(self):
        msg=String()
        msg.data='backup'
        self.publisher.publish(msg)
        self.get_logger().info(f'Send Message:{msg.data}')
def main(args=None):
    rclpy.init(args=args)
    node=Publisher('publisher_test')
    rclpy.spin(node=node)
    rclpy.shutdown()