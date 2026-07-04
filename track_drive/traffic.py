import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


PHASE_3_DURATION = 20.0  # 초반에 3구 신호등만 보는 시간(초)


class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')

        self.bridge = CvBridge()

        # 마우스 콜백용 원본 프레임 저장 + 콜백 등록 여부
        self.frame = None
        self.mouse_cb_set = False
        self.click_point = None   # 마지막 클릭 좌표 (계속 표시용)

        # 3구 -> 4구 전환 타이머
        self.start_time = None
        self.phase_3_done = False

        self.state_pub = self.create_publisher(String, '/state', 10)
        self.detected_pub = self.create_publisher(Bool, '/traffic_light_detected', 10)

        self.sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 마우스 콜백에서 쓸 원본 프레임 저장
        self.frame = frame.copy()

        # 프로그램 시작(첫 프레임) 기준으로 타이머 시작
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("===== 시작, 3구 단계 타이머 시작 =====")

        roi_3, roi_4 = self.set_rois(frame)

        status_3 = "none"
        status_4 = "none"
        type_3 = "none"
        type_4 = "none"
        box_4 = None

        if not self.phase_3_done:
            # 초반: 3구 신호등만 본다
            phase = "3"
            type_3, box_3 = self.detect_traffic_box(roi_3, "3")

            if box_3 is not None:
                status_3 = self.check_3_light(roi_3)

            # 타이머 경과 시 4구 단계로 전환
            elapsed = time.time() - self.start_time
            if elapsed >= PHASE_3_DURATION:
                self.phase_3_done = True
                self.get_logger().info("===== 4구 신호등 단계로 전환 =====")
        else:
            # 18초 경과 후: 4구 신호등만 본다
            phase = "4"
            type_4, box_4 = self.detect_traffic_box(roi_4, "4")

            if box_4 is not None:
                status_4 = self.check_4_light(roi_4)

            # 4구 신호등 감지 여부 publish
            # ── [수정] 박스만 잡혀도 색(status_4)이 none 이면 감지 안 된 것으로 처리.
            #    박스 모양은 잡혔지만 불 색을 확정 못 한 프레임에서 detect=True 가
            #    나가던 문제를 막는다. (색 확정 == status_4 != "none")
            detected_msg = Bool()
            detected_msg.data = (box_4 is not None) and (status_4 != "none")
            self.detected_pub.publish(detected_msg)

        final_status = self.select_final_status(status_3, status_4)

        state_str = self.make_state_string(status_3, status_4)
        if state_str:
            self.state_pub.publish(String(data=state_str))

        self.get_logger().info(
            f"[phase={phase}] "
            f"ROI_3 type={type_3}, status={status_3} | "
            f"ROI_4 type={type_4}, status={status_4} | "
            f"FINAL={final_status} | PUB={state_str}"
        )

        cv2.putText(
            frame,
            f"FINAL: {final_status}  PUB: {state_str}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        # 마지막 클릭 좌표를 매 프레임 다시 그려서 계속 표시
        if self.click_point is not None:
            cx, cy = self.click_point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"({cx},{cy})",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        key = cv2.waitKey(1)
        if key == 27:
            rclpy.shutdown()

    def set_rois(self, frame):
        # 3구 ROI
        x1_3, y1_3 = 159, 35
        x2_3, y2_3 = 432, 156

        # 4구 ROI
        x1_4, y1_4 = 81, 58
        x2_4, y2_4 = 416, 149 

        roi_3 = frame[y1_3:y2_3, x1_3:x2_3]
        roi_4 = frame[y1_4:y2_4, x1_4:x2_4]

        cv2.rectangle(frame, (x1_3, y1_3), (x2_3, y2_3), (255, 0, 0), 2)
        cv2.putText(frame, "ROI_3", (x1_3, y1_3 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1_4, y1_4), (x2_4, y2_4), (0, 255, 0), 2)
        cv2.putText(frame, "ROI_4", (x1_4, y1_4 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return roi_3, roi_4

    def detect_traffic_box(self, roi, name):
        #흑백으로 단순히 만든 후 작은 노이즈 없앤 뒤 경계선만 남김
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        #신호등이 사각형으로 잡힐 수 있게끔 외곽선 메움
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        debug = roi.copy()

        best_box = None
        best_area = 0
        best_black_ratio = 0.0

        #윤곽선들마다 신호등 박스인지 검사
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h
            ratio = w / float(h)

            #작은 것들 제외
            if area < 1000:
                continue
            
            #신호등 답지않은 모양 제외
            if ratio < 1.8:
                continue

            box = roi[y:y + h, x:x + w]

            hsv_box = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            v = hsv_box[:, :, 2]

            black_mask = (v < 80)
            black_ratio = np.sum(black_mask) / float(w * h)

            #신호등 내부처럼 검은색 많은지 검사
            if black_ratio < 0.25:
                continue

            #후보들 중 가장 신호등에 가까운 것 선택
            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)
                best_black_ratio = black_ratio

        #후보가 없다면 검출 안함
        if best_box is None:
            return "none", None

        x, y, w, h = best_box
        ratio = w / float(h)

        #가로 세로 비율을 통해 3구 신호등과 4구 신호등 판별
        if ratio >= 3.3:
            signal_type = "4_light"
        elif ratio >= 2.0:
            signal_type = "3_light"
        else:
            signal_type = "none"

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.putText(
            debug,
            f"{signal_type} r={ratio:.2f} black={best_black_ratio:.2f}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2
        )

        return signal_type, best_box

    #각 색깔부분들 마스크 설정
    def make_color_masks(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        red_mask = (
            (((h >= 0) & (h <= 10)) | ((h >= 170) & (h <= 180)))
            & (s > 80)
            & (v > 80)
        )

        yellow_mask = (
            ((h >= 15) & (h <= 35))
            & (s > 80)
            & (v > 80)
        )

        green_mask = (
            ((h >= 45) & (h <= 80))
            & (s > 80)
            & (v > 80)
        )

        return red_mask, yellow_mask, green_mask

    #색별된 색들 중 가장 픽셀 수가 많은 색을 선택해서 신호등 판별
    def check_3_light(self, roi):
        red_mask, yellow_mask, green_mask = self.make_color_masks(roi)

        height, width = roi.shape[:2]

        a = width // 3
        b = width * 2 // 3

        red_count = np.sum(red_mask[:, 0:a])
        yellow_count = np.sum(yellow_mask[:, a:b])
        green_count = np.sum(green_mask[:, b:width])

        self.get_logger().info(
            f"3구 count | R={red_count}, Y={yellow_count}, G={green_count}"
        )

        counts = {
            "red": red_count,   
            "yellow": yellow_count,
            "green": green_count
        }

        status = max(counts, key=counts.get)

        #가장 많이 나온 색깔의 픽셀값이 일정 이하면 none으로 취급
        if counts[status] < 600:
            return "none"

        return status

    #식별된 색들 중 가장 많이 판별된 픽셀 수로 신호등 판별
    def check_4_light(self, roi):
        red_mask, yellow_mask, green_mask = self.make_color_masks(roi)

        height, width = roi.shape[:2]

        q1 = width // 4
        q2 = width // 2
        q3 = width * 3 // 4

        # 4구 순서: [red] [yellow] [left] [green]
        red_count = np.sum(red_mask[:, 0:q1])
        yellow_count = np.sum(yellow_mask[:, q1:q2])
        left_count = np.sum(green_mask[:, q2:q3])
        green_count = np.sum(green_mask[:, q3:width])

        self.get_logger().info(
            f"4구 count | R={red_count}, Y={yellow_count}, "
            f"L={left_count}, G={green_count}"
        )

        counts = {
            "red": red_count,
            "yellow": yellow_count,
            "left": left_count,
            "green": green_count
        }

        status = max(counts, key=counts.get)

        #가장 많이 판별된 픽셀 수가 일정 이하일 시 none으로 취급
        if counts[status] < 600:
            return "none"

        # ── [수정] 좌회전(left)은 빨간불이 같이 켜져 있을 때만 유효 ──
        #    4구 좌회전 신호 = 빨강(직진 정지) + 좌회전 화살표 동시 점등.
        #    left/green 은 둘 다 같은 green_mask 라, 직진 초록불이 ROI 위치
        #    어긋남으로 left 칸에 새면 4_LEFT 로 오판정됐다. 빨강이 함께
        #    잡히지 않으면 좌회전이 아니라 직진 초록으로 재판정한다.
        if status == "left" and red_count < 600:
            return "green" if green_count >= 600 else "none"

        return status

    def select_final_status(self, status_3, status_4):
        if status_4 == "left":
            return "left"

        if status_4 != "none":
            return status_4

        if status_3 != "none":
            return status_3

        return "none"

    def make_state_string(self, status_3, status_4):
        MAP_3 = {
            "red":    "3_RED",
            "yellow": "3_YELLOW",
            "green":  "3_GREEN",
        }
        MAP_4 = {
            "red":    "4_RED",
            "yellow": "4_YELLOW",
            "green":  "4_GREEN",
            "left":   "4_LEFT",
        }

        if status_4 != "none":
            return MAP_4.get(status_4, "")

        if status_3 != "none":
            return MAP_3.get(status_3, "")

        return ""


def main(args=None):
    rclpy.init(args=args)

    node = ImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()