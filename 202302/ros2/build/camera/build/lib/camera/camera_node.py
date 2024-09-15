import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',  # 图像话题，根据实际设置情况更改
            self.listener_callback,
            10)
        self.publisher1 = self.create_publisher(Image, 'rec_image', 10)

        self.bridge = CvBridge()
        
    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")   #转换为OpenCV图像
        
        # 在图像右上角绘制矩形
        height, width, _ = cv_image.shape
        cv2.rectangle(cv_image, (width - 100, 0), (width, 100), (255, 255, 255), 2)
        
        # 将 OpenCV 图像转换回 ROS 图像消息
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        
        # 发布处理后的图像
        self.publisher1.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

