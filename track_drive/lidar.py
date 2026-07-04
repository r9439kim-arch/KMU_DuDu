#!/usr/bin/env python3

# ==========================================================
# 필요한 라이브러리 import
# - ROS2 통신
# - LiDAR 메시지
# - NumPy 및 수학 연산
# ==========================================================
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import math


# ==========================================================
# LiDAR 기반 대형 차량 검출 노드
#
# 역할
# - LiDAR 데이터 수신
# - 차량 클러스터 생성
# - 차량 여부 판별
# - 차량 중심 좌표 Publish
# ==========================================================
class LidarTracker(Node):
    def __init__(self):
        super().__init__('lidar_tracker')

        self.ranges = None
        self.angle_min = None
        self.angle_increment = None

        self.large_vehicle_detected = False
        self.car_position = None
        self.light4 = False
        self.left_done = False

        self.point_threshold = 19

        self.lost_counter = 0
        self.max_lost_frames = 20
        self.debug_counter = 0

        self.max_cluster_span = 6.0
        self.direction_change_threshold_deg = 35.0

        self.max_planarity_residual = 0.035
        self.corner_detect_angle_deg = 20.0
        self.min_curvature_radius = 3.0

        self.left_rot = False

        # Subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.light_sub = self.create_subscription(
            Bool,
            '/traffic_light_detected',
            self.camera_callback,
            10
        )

        self.left_sub = self.create_subscription(
            Bool, 
            '/left_turn_done', 
            self.left_callback,
            10
        )

        self.car_on = self.create_subscription(
            Bool,
            '/left_rot',
            self.car_on_callback,
            10
        )


        # Publishers
        self.mid_pub = self.create_publisher(
            Point,
            '/mid_point_xy',
            qos_profile_sensor_data
        )

        self.car_pub = self.create_publisher(
            Bool,
            '/large_vehicle_detected',
            10
        )

        self.left_pub = self.create_publisher(
            Bool,
            '/left_obstacles',
            10
        )

        self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Lidar Tracker Started")

    # LiDAR값들 저장
    def lidar_callback(self, msg):
        self.ranges = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
    
    # 4구 신호등 인식 여부 확인
    def camera_callback(self, msg):
        self.light4 = msg.data

    # 좌회전 완료 여부 확인
    def left_callback(self, msg):
        self.left_done = msg.data

    # 지름길 진입 상태 확인
    def car_on_callback(self, msg):
        self.left_rot = msg.data

    # ==========================================================
    # 거리 기반 클러스터 생성
    #
    # 인접한 LiDAR 점들을 하나의 물체로 묶고
    # 최대 크기를 초과하면 새로운 클러스터 생성
    # ==========================================================
    def make_clusters(self, points):
        clusters = []

        if len(points) == 0:
            return clusters

        current_cluster = [points[0]]

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]

            dist = np.linalg.norm(
                np.array(curr) - np.array(prev)
            )

            # 새 점을 추가했을 때 클러스터 전체 크기 계산
            candidate = current_cluster + [curr]
            cand_arr = np.array(candidate)
            span = np.linalg.norm(
                cand_arr.max(axis=0) - cand_arr.min(axis=0)
            )

            if dist < 0.4 and span <= self.max_cluster_span:
                current_cluster.append(curr)

            else:
                if len(current_cluster) >= 5:
                    clusters.append(current_cluster)

                current_cluster = [curr]

        if len(current_cluster) >= 5:
            clusters.append(current_cluster)

        return clusters

    # ==========================================================
    # 방향 변화 기반 클러스터 재분할
    #
    # 하나의 클러스터 안에 여러 물체가 포함된 경우
    # 방향 변화가 큰 지점을 기준으로 분리
    # ==========================================================
    def split_by_direction_change(self, cluster):
        if len(cluster) < 6:
            return [cluster]

        arr = np.array(cluster)
        vectors = np.diff(arr, axis=0)

        split_indices = []

        for i in range(1, len(vectors)):
            v1 = vectors[i - 1]
            v2 = vectors[i]

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < 1e-3 or n2 < 1e-3:
                continue

            cos_angle = np.dot(v1, v2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg > self.direction_change_threshold_deg:
                split_indices.append(i + 1)

        # 방향 변화가 2회 이상이면 하나의 클러스터가 여러 물체일 가능성이 높음
        if len(split_indices) <= 1:
            return [cluster]

        segments = []
        prev_idx = 0

        for sp in split_indices:
            segments.append(cluster[prev_idx:sp])
            prev_idx = sp

        segments.append(cluster[prev_idx:])

        return [seg for seg in segments if len(seg) >= 5]

    # ==========================================================
    # 직선 적합 오차 및 곡률 계산
    #
    # SVD를 이용하여 직선 적합 정도와
    # 곡률 반지름을 계산
    # ==========================================================
    def line_fit_residual_and_curvature(self, points):
        points = np.array(points)

        if len(points) < 4:
            return 0.0, float('inf')

        mean = np.mean(points, axis=0)
        centered = points - mean

        # SVD를 이용해 클러스터의 주축(직선 방향) 계산
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])

        t = centered @ direction
        perp = centered @ normal

        # 직선에서 벗어난 정도(평면성)
        residual = np.std(perp)

        if np.std(t) > 1e-6:
            # 2차 곡선으로 곡률 반지름 추정
            coeffs = np.polyfit(t, perp, 2)
            a = coeffs[0]

            if abs(a) > 1e-6:
                radius = abs(1.0 / (2.0 * a))
            else:
                radius = float('inf')
        else:
            radius = float('inf')

        return residual, radius

    # ==========================================================
    # 코너(ㄱ자) 구조 탐색
    #
    # 진행 방향이 가장 크게 변하는 지점을
    # 차량의 코너 후보로 선택
    # ==========================================================
    def find_corner_index(self, cluster):
        arr = np.array(cluster)

        if len(arr) < 6:
            return None

        vectors = np.diff(arr, axis=0)

        best_idx = None
        best_angle = 0.0

        for i in range(1, len(vectors)):
            v1, v2 = vectors[i - 1], vectors[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if n1 < 1e-3 or n2 < 1e-3:
                continue

            cos_angle = np.dot(v1, v2) / (n1 * n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if angle_deg > best_angle:
                best_angle = angle_deg
                best_idx = i + 1

        if best_angle < self.corner_detect_angle_deg:
            return None

        return best_idx

    # ==========================================================
    # 평면성 검사
    #
    # 직선 차량과 ㄱ자 차량에 대해
    # 직선 적합 오차와 곡률 조건을 확인
    # ==========================================================
    def check_planarity(self, cluster, line_like, corner_like):
        if line_like:
            residual, radius = self.line_fit_residual_and_curvature(cluster)
            return (
                residual < self.max_planarity_residual and
                radius > self.min_curvature_radius
            )

        if corner_like:
            corner_idx = self.find_corner_index(cluster)

            if corner_idx is None:
                residual, radius = self.line_fit_residual_and_curvature(cluster)
                return (
                    residual < self.max_planarity_residual and
                    radius > self.min_curvature_radius
                )

            # ㄱ자 차량은 두 직선으로 나누어 각각 검사
            seg1 = cluster[:corner_idx]
            seg2 = cluster[corner_idx:]

            if len(seg1) < 3 or len(seg2) < 3:
                return False

            residual1, radius1 = self.line_fit_residual_and_curvature(seg1)
            residual2, radius2 = self.line_fit_residual_and_curvature(seg2)

            return (
                residual1 < self.max_planarity_residual and
                residual2 < self.max_planarity_residual and
                radius1 > self.min_curvature_radius and
                radius2 > self.min_curvature_radius
            )

        return False

    # ==========================================================
    # PCA 기반 형상 분석
    #
    # 직선 차량 또는 ㄱ자 차량 여부를 판단하고
    # 평면성 조건을 함께 검사
    # ==========================================================
    def check_shape(self, cluster):
        cluster = np.array(cluster)

        if len(cluster) < 5:
            return False, 0.0

        mean = np.mean(cluster, axis=0)
        centered = cluster - mean

        cov = np.cov(centered.T)

        eigenvalues, _ = np.linalg.eig(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # PCA 기반 직선성(Linearity) 계산
        linearity = (eigenvalues[0] / (eigenvalues[1] + 1e-6))

        line_like = (linearity > 6.0)
        corner_like = (2.0 < linearity < 6.0)

        planarity_valid = self.check_planarity(cluster, line_like, corner_like)
        shape_valid = (line_like or corner_like) and planarity_valid

        return shape_valid, linearity

    # ==========================================================
    # 대형 차량 판별
    #
    # 조건
    # - 위치
    # - 크기
    # - Point 개수
    # - PCA 형상
    # - 평면성
    # ==========================================================
    def is_large_vehicle(self, cluster):
        cluster = np.array(cluster)

        x_min = np.min(cluster[:, 0])
        x_max = np.max(cluster[:, 0])

        y_min = np.min(cluster[:, 1])
        y_max = np.max(cluster[:, 1])

        width = x_max - x_min
        length = y_max - y_min

        point_count = len(cluster)

        # 차량 중심 계산
        cx = np.mean(cluster[:, 0])
        cy = np.mean(cluster[:, 1])

        shape_valid, linearity = self.check_shape(cluster)

        center_condition = (abs(cx) < 2.5)
        distance_condition = (0.0 < cy < 8.0)
        size_condition = (width > 0.8 or length > 0.8)
        corner_size_condition = (width > 0.8 and length > 0.8)
        point_condition = (point_count > 6)

        # 직선 차량 조건
        line_vehicle = (
            linearity > 6.0 and
            shape_valid and
            center_condition and
            distance_condition and
            size_condition and
            point_condition
        )

        # ㄱ자 차량 조건
        corner_vehicle = (
            2.0 < linearity < 6.0 and
            shape_valid and
            center_condition and
            distance_condition and
            corner_size_condition and
            point_condition
        )

        # 최종 판별
        is_vehicle = (line_vehicle or corner_vehicle)

        if is_vehicle:
            return True, (cx, cy), (width, length)

        return False, None, None


    # ==========================================================
    # 차량 검출 결과 Publish
    # ==========================================================
    def update_detection(self, detected):
        if self.large_vehicle_detected != detected:
            self.large_vehicle_detected = detected

        car_msg = Bool()
        car_msg.data = detected
        self.car_pub.publish(car_msg)

    # ==========================================================
    # 메인 처리 루프
    #
    # 처리 순서
    # 1. LiDAR 전처리
    # 2. ROI 추출
    # 3. 좌측 장애물 확인
    # 4. 차량 클러스터 생성
    # 5. 차량 판별
    # 6. 차량 위치 갱신
    # 7. Detection 및 중심 좌표 Publish
    # ==========================================================
    def timer_callback(self):
        if self.ranges is None:
            return

        valid = np.array([
            d if math.isfinite(d)
            else np.nan
            for d in self.ranges
        ])

        angles = (
            self.angle_min +
            np.arange(len(valid)) *
            self.angle_increment
        )

        # Polar → Cartesian 좌표 변환
        x = -valid * np.sin(angles)
        y = valid * np.cos(angles)

        overtake_roi = []
        left_roi = []
        left_count = 0

        # 차량 검출 ROI / 좌측 장애물 ROI 추출
        for px, py, dist in zip(x, y, valid):

            if not math.isfinite(dist):
                continue

            # ROI
            # 차량 감지 영역
            if (py > 0 and py < 6.0 and abs(px) < 4.0):
                overtake_roi.append([px, py])

            # 좌측 경찰차 감지 영역
            if (py > 0.0 and py < 12.5 and px > -10.0 and px < 0.0):
                left_count += 1
                left_roi.append([px, py])



        # 신호등 모드에서는 좌측 장애물만 검사
        if self.light4:
            left_obs = left_count > self.point_threshold

            # Publish
            obs_msg = Bool()
            obs_msg.data = left_obs

            self.left_pub.publish(obs_msg)
            self.get_logger().info(
                f"ROI Count: {left_count}, Detected: {left_obs}" # 장애물이 있으면 True, 아니면 False
            )

            if self.left_done:
                self.left_done = False


        # 지름길에 들어갔다면
        elif self.left_done:
            if self.left_rot:
                self.left_done = False

        # 일반 주행에서는 차량 검출 수행
        else:
            if len(overtake_roi) == 0:
                clusters = []
            else:
                overtake_roi = np.array(overtake_roi)
                clusters = self.make_clusters(overtake_roi.tolist())

                # 방향 변화를 이용해 붙어 있는 클러스터 분리
                refined_clusters = []
                for c in clusters:
                    refined_clusters.extend(self.split_by_direction_change(c))

                clusters = refined_clusters

            car_detect = False
            best_dist = float('inf')

            for cluster in clusters:
                is_vehicle, center, size = \
                    self.is_large_vehicle(cluster)

                if not is_vehicle:
                    continue

                car_detect = True
                cx, cy = center
                dist_to_robot = abs(cy)

                # 가장 가까운 차량
                if dist_to_robot < best_dist:
                    best_dist = dist_to_robot
                    self.car_position = (cx, cy)

            # 차량이 잠시 사라져도 검출 유지
            if car_detect:
                self.lost_counter = 0

            else:
                self.lost_counter += 1

                if self.lost_counter > self.max_lost_frames:
                    self.car_position = None

            final_detected = (self.car_position is not None)

            # Publish Detection
            self.update_detection(final_detected)

            # Publish Center Point
            if (final_detected and self.car_position is not None):
                point_msg = Point()

                point_msg.x = float(self.car_position[0])
                point_msg.y = float(self.car_position[1])
                point_msg.z = 0.0
                self.mid_pub.publish(point_msg)


            # DEBUG
            self.debug_counter += 1

            if self.debug_counter >= 10:
                self.debug_counter = 0

                # 차량 중심 좌표 Publish
                if (final_detected and self.car_position is not None):
                    cx, cy = self.car_position

                    self.get_logger().info(
                        f"[DETECTED] "
                        f"x={cx:.2f}, "
                        f"y={cy:.2f}"
                    )

                else:
                    self.get_logger().info("[NO VEHICLE]")


# Main
def main(args=None):
    rclpy.init(args=args)
    node = LidarTracker()
    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()