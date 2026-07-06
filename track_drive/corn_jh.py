#!/usr/bin/env python3
"""
라바콘 사이 goal_point 발행 노드 (ROS2)

좌표계:
  x = -r * sin(angle)   좌(-) / 우(+)
  y =  r * cos(angle)   후(-) / 전(+)
  원점: 차량(라이다) 위치

ROI: 부채꼴(거리+각도)이 아닌 직사각형(x/y 범위) 필터 사용
  전방: ROI_Y_MIN ~ ROI_Y_MAX
  좌우: ROI_X_MIN ~ ROI_X_MAX

좌/우 레인 분류 로직:
  1. 콘들을 원점에서 가까운 순으로 정렬
  2. 좌/우 레인이 모두 비어있을 때 (초기 상태):
       콘의 x 부호로 분류  (x<0 → 좌, x>=0 → 우)
  3. 이미 한쪽 레인이라도 콘이 있을 때:
       각 레인의 anchor(절대 출발점, 고정)와의 거리로 판단해
       더 가까운 레인에 배정. 둘 다 ANCHOR_MAX_DIST를 넘으면
       나무 등 배경 물체로 보고 버림.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np


# ===================== 파라미터 튜닝 영역 =====================

# 라이다 ROI 필터 (직사각형, 차량 기준 상대 좌표)
ROI_Y_MIN = 0.3      # 전방 최소 거리 (m) — 차체 자체 반사 노이즈 제거
ROI_Y_MAX = 7.0      # 전방 최대 거리 (m) — 트랙 크기에 맞게 축소 (배경 나무/신호등/벽 제외)
ROI_X_MIN = -4.5     # 좌측 최대 폭 (m)
ROI_X_MAX =  4.5     # 우측 최대 폭 (m)

# DBSCAN
DBSCAN_EPS         = 0.3   # 같은 클러스터로 묶을 최대 거리 (m) — 콘 한 개 폭 정도로 축소
DBSCAN_MIN_SAMPLES = 2

# 콘 포인트 개수 필터 (라바콘은 라이다에 적은 포인트만 찍힘)
CONE_MIN_POINTS = 2    # 너무 적으면(1개) 노이즈로 간주해 제외
CONE_MAX_POINTS = 4    # 너무 많으면(나무, 기둥, 벽 등 큰 물체) 제외

# (콘 유효 범위는 위 ROI_X/Y 필터가 이미 처리)

# 좌/우 레인 분류 — anchor(절대 출발점) 기준 거리로 판단
ANCHOR_MAX_DIST = 3.0   # 같은 레인 anchor와 이 거리 이내인 콘만 그 레인으로 인정 (m)
                         # 둘 다 이 거리를 넘으면 나무 등 배경 물체로 보고 버림

# 쌍 매칭: 좌/우 콘의 y값(전방 거리) 차이 허용 범위
MAX_PAIR_Y_DIFF = 1.5   # (m)

# goal_point 발행
GOAL_PUBLISH_HZ  = 10.0
GOAL_ADVANCE_IDX = 1      # 몇 번째 앞 중앙점을 goal로 쓸지

# 한쪽만 보일 때 중앙 추정 오프셋
ONE_SIDE_OFFSET = 2.8   # (m) — 대표 트랙 폭의 절반

# =============================================================


def dbscan_numpy(points, eps, min_samples):
    n = len(points)
    labels  = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    diff     = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=2))

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist_mat[i] <= eps)[0].tolist()

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed = neighbors[:]
        j = 0
        while j < len(seed):
            s = seed[j]
            if not visited[s]:
                visited[s] = True
                new_nb = np.where(dist_mat[s] <= eps)[0].tolist()
                if len(new_nb) >= min_samples:
                    seed.extend(new_nb)
            if labels[s] == -1:
                labels[s] = cluster_id
            j += 1

        cluster_id += 1

    return labels


class ConeGoalPointNode(Node):
    def __init__(self):
        super().__init__('cone_goal_point_node')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.goal_pub   = self.create_publisher(Point,       '/goal_point',   10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_markers', 10)

        self.cone_pub = self.create_publisher(Bool, '/cone_detect', 10)

        self.goal_path = []
        self.scan_frame_id = 'laser'   # scan 콜백에서 실제 값으로 갱신됨

        self.create_timer(1.0 / GOAL_PUBLISH_HZ, self.publish_goal)
        self.get_logger().info('🟠 라바콘 goal_point 노드 시작!')

    # ──────────────────────────────────────────
    # 1. LaserScan → XY
    # ──────────────────────────────────────────
    def laserscan_to_xy(self, scan):
        ranges = np.array(scan.ranges, dtype=float)
        angles = np.linspace(0, 2 * np.pi, len(self.ranges)) - np.pi / 2
        ranges = np.where(np.isfinite(ranges), ranges, np.nan)

        finite = np.isfinite(ranges)
        r, a = ranges[finite], angles[finite]

        x = r * np.cos(a) * 100
        y = r * np.sin(a) * 100

        # 전방: ROI_Y_MIN ~ ROI_Y_MAX,  좌우: ROI_X_MIN ~ ROI_X_MAX
        valid = (
            (y >= ROI_Y_MIN) & (y <= ROI_Y_MAX) &
            (x >= ROI_X_MIN) & (x <= ROI_X_MAX)
        )

        return np.column_stack((x[valid], y[valid]))

    # ──────────────────────────────────────────
    # 2. DBSCAN → 콘 중심 리스트
    # ──────────────────────────────────────────

    def detect_cones(self, points):
        if len(points) < DBSCAN_MIN_SAMPLES:
            return []

        labels = dbscan_numpy(points, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

        cones = []
        for lbl in set(labels):
            if lbl == -1:
                continue
            cluster = points[labels == lbl]
            n_points = len(cluster)

            # 콘은 포인트가 적게 찍힌다 — 너무 적으면 노이즈, 너무 많으면 큰 물체(나무/기둥/벽)
            if n_points < CONE_MIN_POINTS or n_points > CONE_MAX_POINTS:
                continue

            c = cluster.mean(axis=0)
            cones.append(c)   # ROI 필터는 laserscan_to_xy에서 이미 적용됨

        return cones

    # ──────────────────────────────────────────
    #
    #    원점에서 가까운 콘부터 순서대로 하나씩 배정한다.
    #
    #    - 양쪽 레인이 모두 비어있을 때(첫 콘):
    #         x < 0 → 좌측 anchor(절대 출발점) 지정
    #         x >= 0 → 우측 anchor 지정
    #
    #    - 한쪽 레인만 비어있을 때:
    #         있는 레인의 anchor(고정)와의 거리를 계산.
    #         거리가 ANCHOR_MAX_DIST 이내면 그 레인에 추가.
    #         아니면서 x부호가 반대쪽이면 반대편 레인의 새 anchor로.
    #         양쪽 다 아니면(너무 멀고 방향도 애매) → 노이즈로 버림.
    #
    #    - 양쪽 레인 모두 존재할 때:
    #         각 레인의 anchor(고정)와의 거리를 비교.
    #         둘 다 ANCHOR_MAX_DIST를 넘으면 → 노이즈(나무 등)로 버림.
    #         (먼 물체를 억지로 한쪽 레인에 끼워넣지 않는 게 핵심)
    #         하나만 범위 안이면 그 레인에, 둘 다 범위 안이면 더 가까운 쪽에 배정.
    # ──────────────────────────────────────────
    def lane_score(self, anchor_cone, candidate):
        return math.hypot(candidate[0] - anchor_cone[0],
                           candidate[1] - anchor_cone[1])

    def classify_cones(self, cones):
        if not cones:
            return [], []

        # 원점에서 가까운 순으로 처리
        order = np.argsort([math.hypot(c[0], c[1]) for c in cones])

        left = []
        right = []
        left_anchor = None   # 좌측 레인의 절대 출발점 (불변)
        right_anchor = None   # 우측 레인의 절대 출발점 (불변)

        for idx in order:
            c = cones[idx]

            # ── 케이스 1: 양쪽 다 비어있음 → x 부호로 시작점(anchor) 결정
            if left_anchor is None and right_anchor is None:
                if c[0] < 0:
                    left.append(c)
                    left_anchor = c
                else:
                    right.append(c)
                    right_anchor = c
                continue

            # ── 케이스 2: 한쪽만 존재
            if left_anchor is not None and right_anchor is None:
                score = self.lane_score(left_anchor, c)
                if score <= ANCHOR_MAX_DIST:
                    left.append(c)
                elif c[0] >= 0:
                    right.append(c)
                    right_anchor = c
                continue

            if right_anchor is not None and left_anchor is None:
                score = self.lane_score(right_anchor, c)
                if score <= ANCHOR_MAX_DIST:
                    right.append(c)
                elif c[0] < 0:
                    left.append(c)
                    left_anchor = c
                continue

            # ── 케이스 3: 양쪽 다 존재
            score_l = self.lane_score(left_anchor, c)
            score_r = self.lane_score(right_anchor, c)

            l_ok = score_l <= ANCHOR_MAX_DIST
            r_ok = score_r <= ANCHOR_MAX_DIST

            if not l_ok and not r_ok:
                continue   # 둘 다 멀다 → 나무/배경 물체로 보고 버림
            elif l_ok and not r_ok:
                left.append(c)
            elif r_ok and not l_ok:
                right.append(c)
            else:
                # 둘 다 범위 안 → 더 가까운 쪽
                if score_l <= score_r:
                    left.append(c)
                else:
                    right.append(c)

        return left, right

    # ──────────────────────────────────────────
    # 4. 좌/우 쌍 매칭 → 중앙점 경로
    #    y값(전방 거리)이 비슷한 쌍끼리 매칭
    # ──────────────────────────────────────────

    def build_midpoint_path(self, left, right):
        if not left and not right:
            return []

        # 한쪽만 보일 때: 보이는 콘 기준 오프셋
        if not right:
            return [np.array([c[0] + ONE_SIDE_OFFSET, c[1]])
                    for c in sorted(left, key=lambda c: c[1])]
        if not left:
            return [np.array([c[0] - ONE_SIDE_OFFSET, c[1]])
                    for c in sorted(right, key=lambda c: c[1])]

        left_sorted  = sorted(left,  key=lambda c: c[1])
        right_sorted = sorted(right, key=lambda c: c[1])

        midpoints  = []
        used_right = set()

        for lc in left_sorted:
            best_idx, best_diff = None, float('inf')
            for i, rc in enumerate(right_sorted):
                if i in used_right:
                    continue
                y_diff = abs(lc[1] - rc[1])
                if y_diff < MAX_PAIR_Y_DIFF and y_diff < best_diff:
                    best_diff = y_diff
                    best_idx  = i

            if best_idx is not None:
                used_right.add(best_idx)
                mid = (lc + right_sorted[best_idx]) / 2.0
                midpoints.append(mid)

        midpoints.sort(key=lambda p: p[1])
        return midpoints

    # ──────────────────────────────────────────
    # 5. scan 콜백
    # ──────────────────────────────────────────

    def scan_callback(self, scan: LaserScan):
        self.scan_frame_id = scan.header.frame_id or self.scan_frame_id
        points = self.laserscan_to_xy(scan)
        cones  = self.detect_cones(points)
        left, right = self.classify_cones(cones)
        path   = self.build_midpoint_path(left, right)

        if path:
            self.goal_path = path
            self.get_logger().info(
                f'콘 {len(cones)}개 (좌:{len(left)} 우:{len(right)}) '
                f'→ 중앙점 {len(path)}개',
                throttle_duration_sec=0.5
            )
            left_str  = ', '.join(f'({c[0]:.1f},{c[1]:.1f})' for c in left)
            right_str = ', '.join(f'({c[0]:.1f},{c[1]:.1f})' for c in right)
            self.get_logger().info(
                f'  좌측: [{left_str}]  /  우측: [{right_str}]',
                throttle_duration_sec=0.5
            )
        else:
            self.get_logger().warn('콘 쌍 없음 - 이전 경로 유지',
                                   throttle_duration_sec=1.0)

        self.publish_markers(left, right, path)

    # ──────────────────────────────────────────
    # 6. goal_point 발행 타이머
    # ──────────────────────────────────────────
    
    def publish_goal(self):
        if not self.goal_path:
            msg_detect = Bool()
            msg_detect.data = False
            self.cone_pub.publish(msg_detect)
            return

        idx    = min(GOAL_ADVANCE_IDX, len(self.goal_path) - 1)
        target = self.goal_path[idx]

        msg_detect = Bool()
        msg_detect.data = True
        self.cone_pub.publish(msg_detect)

        msg   = Point()
        msg.x = float(target[0])
        msg.y = float(target[1])
        msg.z = 0.0
        self.goal_pub.publish(msg)

        self.get_logger().info(
            f'▶ goal_point [{idx+1}/{len(self.goal_path)}] '
            f'x={msg.x:.2f}(좌우), y={msg.y:.2f}(전방)',
            throttle_duration_sec=0.3
        )

    # ──────────────────────────────────────────
    # 7. RViz 마커 시각화
    # ──────────────────────────────────────────
    def publish_markers(self, left, right, midpoints):
        if not self.marker_pub.get_subscription_count():
            return

        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # 이전 프레임의 잔여 마커를 모두 지움 (안 그러면 사라진 콘이 RViz에 남음)
        clear = Marker()
        clear.header.frame_id = self.scan_frame_id
        clear.header.stamp    = stamp
        clear.action           = Marker.DELETEALL
        array.markers.append(clear)

        def make_cone_marker(ns, i, c, r, g, b):
            m = Marker()
            m.header.frame_id = self.scan_frame_id
            m.header.stamp    = stamp
            m.ns, m.id        = ns, i
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
            return m

        for i, c in enumerate(left):   # 좌측 = 파랑
            array.markers.append(make_cone_marker('left_cones',  i, c, 0.0, 0.5, 1.0))
        for i, c in enumerate(right):  # 우측 = 주황
            array.markers.append(make_cone_marker('right_cones', i, c, 1.0, 0.5, 0.0))

        for i, p in enumerate(midpoints):   # 중앙점 = 초록
            array.markers.append(make_cone_marker('midpoints', i, p, 0.0, 1.0, 0.0))

        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = ConeGoalPointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()