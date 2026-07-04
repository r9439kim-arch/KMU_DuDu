import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from collections import deque
import cv2
import numpy as np


# ===== 설정 =====
ROI    = (130, 288, 343, 324)   # (x0, y0, x1, y1)
H_LO, H_HI = 20, 40             # 노란색 H 범위 (HSV=30 중심)
S_LO, V_LO = 120, 120           # 채도/명도 하한 (아스팔트·잔디 배제)
MIN_AREA   = 300                # 노이즈 제거 최소 면적
MIN_W      = 70                # 가로획 최소 폭(px) — 이 값만 튜닝하면 됨
N, M       = 5, 3               # 최근 N프레임 중 M회 이상 검출 시 확정
TOPIC      = '/usb_cam/image_raw/front'


def detect_wide_yellow(bgr, roi,
                       h_lo=H_LO, h_hi=H_HI, s_lo=S_LO, v_lo=V_LO,
                       min_area=MIN_AREA, min_w=MIN_W):
    """
    ROI 안에서 '가로로 넓은 노란 blob' 검출.
    roi: (x0, y0, x1, y1)
    return: (검출여부, [(cx, cy, bbox_global, w), ...], roi_mask)
    """
    x0, y0, x1, y1 = roi
    sub = bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    hits = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        # 가로로 넓으면 통과
        if w >= min_w:                       
            gx, gy = cents[i][0] + x0, cents[i][1] + y0
            hits.append((gx, gy, (x + x0, y + y0, w, h), w))
    return (len(hits) > 0), hits, mask


class TMarkerViewer(Node):
    def __init__(self):
        super().__init__('t_marker_viewer')
        self.bridge = CvBridge()
        self.hist = deque(maxlen=N)
        self.triggered = False

        # 좌회전 완료 신호를 받기 전에는 detect 안 함
        self.left_done = False

        self.sub = self.create_subscription(
            Image, TOPIC, self.callback, 10)

        # 좌회전 완료 신호 구독
        self.left_sub = self.create_subscription(
            Bool, '/left_turn_done', self.left_callback, 10)

        # 지름길 탈출(좌회전 판단) publish
        self.check_pub = self.create_publisher(Bool, '/left_rot', 10)

    def left_callback(self, msg):
        self.left_done = msg.data

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 좌회전 완료 신호(/left_turn_done)를 받기 전에는 detect 안 함
        if not self.left_done:
            return

        found, hits, mask = detect_wide_yellow(frame, ROI)

        self.hist.append(found)
        confirmed = sum(self.hist) >= M

        # 상승엣지에서만 한 번 발동
        if confirmed and not self.triggered:
            self.triggered = True
            widest = max(hits, key=lambda t: t[3]) if hits else None
            self.get_logger().info(f"T-marker TRIGGER  {widest}")
            # 좌회전 트리거 -> /left_rot 에 True publish
            self.check_pub.publish(Bool(data=True))
        elif not confirmed:
            self.triggered = False   # 표식 사라지면 리셋

        # ===== 디버그 화면 =====
        disp = frame.copy()
        x0, y0, x1, y1 = ROI
        col = (0, 255, 0) if found else (0, 0, 255)
        cv2.rectangle(disp, (x0, y0), (x1, y1), col, 2)
        for cx, cy, (bx, by, bw, bh), w in hits:
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            cv2.putText(disp, f"w={w}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        status = "TRIGGER" if self.triggered else ("DETECT" if found else "----")
        cv2.putText(disp, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # cv2.imshow("front", disp)
        # cv2.imshow("roi_mask", mask)
        if cv2.waitKey(1) == 27:   # ESC
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TMarkerViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()