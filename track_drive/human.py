from ultralytics import YOLO
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool

class HumanDetection(Node):
    def __init__(self):
        super().__init__('human_detection')

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.sub_front = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.pub=self.create_publisher(
            Bool,
            '/human_detect',
            10
        )

        self.get_logger().info("Human Detection Node Started")

    def cam_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        h, w = frame.shape[:2]

        roi = frame[:, 30 : (w*3)//5]

        results = self.model(
            roi,
            classes=[0],
            conf=0.8,
            verbose=False
        )

        annotated = results[0].plot()

        human_detected = len(results[0].boxes) > 0

        msg = Bool()
        msg.data = human_detected
        self.pub.publish(msg)

        # cv2.imshow("detection", annotated)
        cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)

    node = HumanDetection()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()