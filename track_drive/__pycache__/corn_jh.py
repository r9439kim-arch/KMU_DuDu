#!/usr/bin/env python3

# ==========================================================
# 필요한 라이브러리 import
# - ROS2 통신
# - 수학 연산
# - NumPy
# ==========================================================
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


# ==========================================================
# 시스템에서 사용하는 모든 파라미터 정의
# - LiDAR 필터링
# - ROI
# - DBSCAN
# - 라바콘 조건
# - 게이트 생성 조건
# - 경로 생성 파라미터
# ==========================================================
class Params:
    # ── 라이다 필터 ──────────────────────────────────────
    MAX_RANGE          = 6.0    # [m] 최대 감지 거리
    MIN_RANGE          = 0.05   # [m] 최소 감지 거리

    # ── 직사각형 ROI (전방=y+, 우측=x+, 좌측=x-) ─────────
    ROI_X_MIN          = -3.0   # [m] 좌측 한계
    ROI_X_MAX          =  3.0   # [m] 우측 한계
    ROI_Y_MIN          =  0.1   # [m] 전방 최소 거리
    ROI_Y_MAX          =  5.0   # [m] 전방 최대 거리

    # ── DBSCAN 클러스터링 ────────────────────────────────
    CLUSTER_EPS        = 0.15   # [m] 같은 클러스터로 볼 최대 거리
    CLUSTER_MIN_PTS    = 2      # 클러스터 최소 포인트 수
    CONE_MAX_WIDTH     = 0.30   # [m] 라바콘 최대 폭
    CONE_MIN_WIDTH     = 0.02   # [m] 라바콘 최소 폭
    CONE_MAX_PTS       = 10     # 클러스터 최대 포인트 수 (그루터기 제거)

    # ── 좌/우 쌍 매칭 ────────────────────────────────────
    GATE_MAX_WIDTH     = 12.0   # [m] 게이트 최대 폭
    GATE_MIN_WIDTH     = 0.8    # [m] 게이트 최소 폭
    PAIR_DEPTH_TOL     = 1.5    # [m] 좌우 콘의 y 차이 허용 오차 (전방 깊이)
    GATE_ANGLE_DIFF_MIN = 0.5   # [rad] 두 콘의 방위각 차이 최소 (~28°)

    # ── 경로 생성 ────────────────────────────────────────
    SHARP_TURN_DEG     = 40.0   # [°] 급회전 판단 각도
    PRE_POST_OFFSET    = 0.35   # [m] 급회전 보조점 거리
    BEZIER_SAMPLES     = 8      # Bezier 보간 샘플 수


# ==========================================================
# LiDAR Polar 좌표를 Cartesian 좌표(x,y)로 변환
# ROI와 거리 조건을 만족하는 점만 추출
# ==========================================================
def polar_to_xy(ranges, angle_min, angle_increment):
    valid = np.array([d if math.isfinite(d) else np.nan for d in ranges])
    angles = angle_min + np.arange(len(valid)) * angle_increment

    x = -valid * np.sin(angles)
    y =  valid * np.cos(angles)

    points = []
    for px, py, dist in zip(x, y, valid):
        if not (math.isfinite(dist) and Params.MIN_RANGE < dist < Params.MAX_RANGE):
            continue
        if not (Params.ROI_X_MIN <= px <= Params.ROI_X_MAX):
            continue
        if not (Params.ROI_Y_MIN <= py <= Params.ROI_Y_MAX):
            continue
        points.append([px, py])

    return np.array(points) if points else np.empty((0, 2))


# ==========================================================
# DBSCAN 알고리즘
# 가까운 점들을 하나의 클러스터로 묶음
# ==========================================================
def dbscan(points, eps, min_pts):
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    def neighbors(idx):
        return np.where(np.linalg.norm(points - points[idx], axis=1) <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if len(nb) < min_pts:
            continue
        labels[i] = cluster_id
        queue = deque(nb)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                nb2 = neighbors(j)
                if len(nb2) >= min_pts:
                    queue.extend(nb2)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels

# 라바콘 후보 중심점 추출
def extract_cone_centers(points):
    if len(points) < 2:
        return []
    labels = dbscan(points, Params.CLUSTER_EPS, Params.CLUSTER_MIN_PTS)
    cones = []
    for cid in set(labels):
        if cid == -1:
            continue
        cluster = points[labels == cid]
        if len(cluster) > Params.CONE_MAX_PTS:
            continue
        w = max(np.max(cluster[:,0])-np.min(cluster[:,0]),
                np.max(cluster[:,1])-np.min(cluster[:,1]))
        if not (Params.CONE_MIN_WIDTH <= w <= Params.CONE_MAX_WIDTH):
            continue
        cx, cy = cluster.mean(axis=0)
        cones.append((float(cx), float(cy)))
    return cones


# ==========================================================
# 좌우 라바콘을 하나의 게이트로 매칭
#
# 조건
# 1. 전방 거리(y)가 비슷할 것
# 2. 좌우에 위치할 것
# 3. 폭이 정상 범위일 것
# ==========================================================
def match_cone_pairs(cones):
    cones_sorted = sorted(range(len(cones)),
                          key=lambda i: math.hypot(cones[i][0], cones[i][1]))
    angles = [math.atan2(c[0], c[1]) for c in cones]

    pairs = []
    used = set()

    for i in cones_sorted:
        if i in used:
            continue
        ci = cones[i]
        best_j, best_score = None, float('inf')

        for j in cones_sorted:
            if j == i or j in used:
                continue
            cj = cones[j]

            y_diff = abs(ci[1] - cj[1])
            if y_diff > Params.PAIR_DEPTH_TOL:
                continue

            adiff = abs(angles[i] - angles[j])
            if adiff > math.pi:
                adiff = 2*math.pi - adiff
            if adiff < Params.GATE_ANGLE_DIFF_MIN:
                continue

            dist = math.hypot(ci[0]-cj[0], ci[1]-cj[1])
            if not (Params.GATE_MIN_WIDTH <= dist <= Params.GATE_MAX_WIDTH):
                continue

            if y_diff < best_score:
                best_score, best_j = y_diff, j

        if best_j is not None:
            cj = cones[best_j]
            if ci[0] <= cj[0]:
                pairs.append((ci, cj))
            else:
                pairs.append((cj, ci))
            used.add(i)
            used.add(best_j)

    pairs.sort(key=lambda p: math.hypot((p[0][0]+p[1][0])/2,
                                         (p[0][1]+p[1][1])/2))
    return pairs

# ==========================================================
# 짝이 없는 라바콘 처리
#
# 1. 다른 단일 콘과 재매칭 시도
# 2. 실패하면 가상의 콘을 생성하여
#    하나의 게이트를 구성
# =========================================================
def estimate_single_cone_pairs(cones, paired_indices, half_gate_width):
    singles = [(i, cones[i]) for i in range(len(cones)) if i not in paired_indices]
    if not singles:
        return []

    result = []
    used = set()

    # 1단계: x 부호 반대끼리 매칭
    for i, (idx_i, ci) in enumerate(singles):
        if idx_i in used:
            continue
        best_j, best_ydiff = None, float('inf')
        for j, (idx_j, cj) in enumerate(singles):
            if idx_j in used or idx_j == idx_i:
                continue
            if ci[0] * cj[0] >= 0:
                continue
            yd = abs(ci[1] - cj[1])
            if yd > 1.5:
                continue
            if yd < best_ydiff:
                best_ydiff, best_j = yd, j
        if best_j is not None:
            _, cj = singles[best_j]
            if ci[0] <= cj[0]:
                result.append((ci, cj))
            else:
                result.append((cj, ci))
            used.add(idx_i)
            used.add(singles[best_j][0])

    # 2단계: 가상 콘 생성
    w = half_gate_width if half_gate_width > 0 else 1.5
    for idx, cone in singles:
        if idx in used:
            continue
        cx, cy = cone
        if cx <= 0:
            virtual = (cx + w*2, cy)
            result.append((cone, virtual))
        else:
            virtual = (cx - w*2, cy)
            result.append((virtual, cone))

    return result


# ==========================================================
# 경로 생성을 위한 보조 함수
#
# midpoint      : 게이트 중앙 계산
# heading_angle : 진행 방향 계산
# ==========================================================
def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

def heading_angle(p_from, p_to):
    return math.atan2(p_to[1]-p_from[1], p_to[0]-p_from[0])

# ==========================================================
# 급격한 방향 전환(Sharp Turn) 탐지
#
# waypoint 사이의 각도를 계산하여
# 일정 각도 이상이면 급회전으로 판단
# ==========================================================
def detect_sharp_turn(wps, threshold_deg=Params.SHARP_TURN_DEG):
    is_sharp = [False]*len(wps)
    for i in range(1, len(wps)-1):
        v1 = (wps[i][0]-wps[i-1][0], wps[i][1]-wps[i-1][1])
        v2 = (wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1])
        m1, m2 = math.hypot(*v1), math.hypot(*v2)
        if m1 < 1e-6 or m2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(m1*m2)))
        if math.degrees(math.acos(cos_a)) > threshold_deg:
            is_sharp[i] = True
    return is_sharp

# ==========================================================
# 급회전 waypoint 앞뒤에 보조점을 삽입
#
# 목적
# - 코너를 부드럽게 통과
# - Bezier 보간 품질 향상
# ==========================================================
def insert_pre_post(wps, is_sharp, offset=Params.PRE_POST_OFFSET):
    result = []
    for i, wp in enumerate(wps):
        if is_sharp[i] and 0 < i < len(wps)-1:
            pre_a  = heading_angle(wp, wps[i-1])
            post_a = heading_angle(wp, wps[i+1])
            result.append((wp[0]+offset*math.cos(pre_a),  wp[1]+offset*math.sin(pre_a)))
            result.append(wp)
            result.append((wp[0]+offset*math.cos(post_a), wp[1]+offset*math.sin(post_a)))
        else:
            result.append(wp)
    return result

# ==========================================================
# Cubic Bezier Curve를 이용하여
# waypoint를 부드러운 곡선으로 변환
# ==========================================================
def bezier_smooth(wps, n=Params.BEZIER_SAMPLES):
    if len(wps) < 2:
        return wps
    pts = np.array(wps)
    smooth = []
    for i in range(len(pts)-1):
        p0 = pts[max(i-1,0)]
        p1 = pts[i]
        p2 = pts[i+1]
        p3 = pts[min(i+2,len(pts)-1)]
        cp1 = p1 + (p2-p0)/6.0
        cp2 = p2 - (p3-p1)/6.0
        for t in np.linspace(0, 1, n, endpoint=(i==len(pts)-2)):
            pt = (1-t)**3*p1 + 3*(1-t)**2*t*cp1 + 3*(1-t)*t**2*cp2 + t**3*p2
            smooth.append(tuple(pt))
    return smooth

# ==========================================================
# 최종 Goal Point 생성
#
# 과정
# 1. 게이트 중앙 계산
# 2. 급회전 탐지
# 3. 보조점 추가
# 4. Bezier 곡선 생성
# 5. 각 waypoint의 heading 계산
# ==========================================================
def build_goal_points(cone_pairs):
    if not cone_pairs:
        return []
    raw       = [midpoint(lc, rc) for lc, rc in cone_pairs]
    is_sharp  = detect_sharp_turn(raw)
    augmented = insert_pre_post(raw, is_sharp)
    smoothed  = bezier_smooth(augmented)

    if len(smoothed) < 2:
        pt = smoothed[0]
        return [(pt[0], pt[1], heading_angle((0.0, 0.0), pt))]

    result = []
    for i, pt in enumerate(smoothed):
        yaw = heading_angle(pt, smoothed[i+1]) if i < len(smoothed)-1 \
              else heading_angle(smoothed[-2], pt)
        result.append((pt[0], pt[1], yaw))
    return result


# ==========================================================
# ROS2 노드
#
# 역할
# - LiDAR 구독
# - Goal Point 생성
# - Path 생성
# ==========================================================
class LidarConeGoalNode(Node):
    def __init__(self):
        super().__init__('lidar_cone_goal_node')

        self.pub_goal  = self.create_publisher(PoseStamped, '/goal_point',     10)
        self.pub_path  = self.create_publisher(Path,        '/lavacone/path',  10)

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.goal_points     = []
        self.cone_pairs      = []
        self.half_gate_width = 0.0

        self.get_logger().info("LidarConeGoalNode 시작 — /scan 대기 중...")

    # ==========================================================
    # LiDAR Scan Callback
    #
    # 처리 순서
    # 1. 좌표 변환
    # 2. 라바콘 검출
    # 3. 게이트 매칭
    # 4. 단일 콘 처리
    # 5. Goal Point 생성
    # ==========================================================    
    def scan_callback(self, msg: LaserScan):
        points = polar_to_xy(np.array(msg.ranges),
                             msg.angle_min, msg.angle_increment)
        if len(points) == 0:
            return

        cones = extract_cone_centers(points)
        if not cones:
            return

        # 정상 매칭
        pairs = match_cone_pairs(cones)

        # 반폭 갱신 + paired 추적
        paired_indices = set()
        if pairs:
            widths = []
            for lc, rc in pairs:
                widths.append(abs(lc[0] - rc[0]) / 2.0)
                for i, c in enumerate(cones):
                    if c == lc or c == rc:
                        paired_indices.add(i)
            self.half_gate_width = sum(widths) / len(widths)

        # 단일 콘 처리
        single_pairs = estimate_single_cone_pairs(
            cones, paired_indices, self.half_gate_width)

        all_pairs = pairs + single_pairs
        if not all_pairs:
            return

        all_pairs.sort(key=lambda p: math.hypot((p[0][0]+p[1][0])/2,
                                                 (p[0][1]+p[1][1])/2))

        new_goals = build_goal_points(all_pairs)
        if new_goals:
            self.cone_pairs  = all_pairs
            self.goal_points = new_goals

        self._publish_path()
        self._publish_current_goal()

    # ==========================================================
    # 현재 목표점(PoseStamped) Publish
    # ==========================================================    
    def _publish_current_goal(self):
        if not self.goal_points:
            return
        x, y, yaw = self.goal_points[0]
        self.get_logger().info(
            f"goal x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.1f}° "
            f"(게이트 {len(self.cone_pairs)}개)")

    # ==========================================================
    # 전체 경로(Path) Publish
    # ==========================================================
    def _publish_path(self):
        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = 'laser'
        path.poses = []
        self.pub_path.publish(path)



# ==========================================================
#  엔트리포인트
# ==========================================================
def main(args=None):
    rclpy.init(args=args)
    node = LidarConeGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()