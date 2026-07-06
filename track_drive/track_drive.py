import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from std_msgs.msg import String
import numpy as np
from std_msgs.msg import Float32MultiArray
# from xycar_msgs.msg import XycarMotor
from nav_msgs.msg import Path
from track_drive.pp_jm import PurePursuit


class State(Node):
    def __init__(self):
        super().__init__('state_node')

        self.state_pub = self.create_publisher(Float32MultiArray, "/xycar_motor", 10)
        self.car_pub   = self.create_publisher(Bool, '/car_lane', 10)
        self.left_done_pub = self.create_publisher(Bool, '/left_turn_done', 10)

        # ── Subscriber ─────────────────────────────────────────────────
        self.lane_bool_sub = self.create_subscription(
            Bool, 'lane_bool', self.lane_bool_callback, 10)

        self.lane_steer_sub = self.create_subscription(
            Float32, '/lane_steer', self.state_callback, 10)
        
        self.lane_curve_score_sub = self.create_subscription(
            Float32, '/lane_curve_score', self.curve_score_callback, 10)
        
        self.lane_center_error_sub = self.create_subscription(
            Float32, '/lane_center_error', self.center_error_callback, 10)

        self.lane_child_sub = self.create_subscription(
            Bool, '/lane_child', self.child_callback, 10)

        self.blinker_sub = self.create_subscription(
            String, '/state', self.blinker_callback, 10)

        self.path_sub = self.create_subscription(
            Path, '/lavacone/path', self.path_callback, 10)

        self.human_sub = self.create_subscription(
            Bool, '/human_detect', self.human_callback, 10)

        self.left_obstacles_sub = self.create_subscription(
            Bool, '/left_obstacles', self.left_obstacles_callback, 10)

        self.left_rot_sub = self.create_subscription(
            Bool, '/left_rot', self.left_rot_callback, 10)

        self.traffic_detected_sub = self.create_subscription(
            Bool, '/traffic_light_detected', self.traffic_detected_callback, 10)

        self.car_sub = self.create_subscription(
            Bool, '/large_vehicle_detected', self.car_detect, 10)

        # ── [추가] 추월 관련 구독 ──────────────────────────────────────
        # lane 코드가 CHANGE/CHASE 중일 때 True 송출
        self.overtake_active_sub = self.create_subscription(
            Bool, '/overtake_active', self.overtake_active_callback, 10)

        # lane 코드 FSM이 계산한 목표 속도 (긴급정지 반영 완료)
        self.overtake_speed_sub = self.create_subscription(
            Float32, '/overtake_speed', self.overtake_speed_callback, 10)

        # ── 상태 변수 ──────────────────────────────────────────────────
        self.pp = PurePursuit()
        self.pp_steer = 0.0
        self.steer    = 0.0
        self.speed    = 0.0
        self.child    = False
        self.traffic_mode  = True
        self.signal        = "LANE"
        self.pp_mode       = False
        self.lane_timer    = None
        self.human_detected = False
        self.left_turn_time = None

        self.corner_triggered  = False
        self.corner_steer      = 0.0
        self.corner_start_time = None
        self.corner_done       = False
        self.CORNER_HOLD_TIME  = 0.95
        self.CORNER_TRIGGER    = 53.0

        self.left_obstacles = False
        self.left_rot       = False
        self.left_rot_doing = False   # 지름길 좌회전 실행 중 (5초 잠금)
        self.left_rot_used = False    # 지름길 좌회전 이미 했는지 (한 번만)
        self.left_done      = False
        self.left_turning = False   
        self.count          = 0
        self.left_rot_start_time = None
        self.cone_end       = False
        self.traffic_light_detected = False
        self.car            = False

        # ── [추가] 추월 상태 변수 ──────────────────────────────────────
        self.overtake_active = False   # lane FSM이 CHANGE/CHASE 중
        self.overtake_speed  = 0.0    # lane FSM이 원하는 속도

        self.curve_score = 99.0
        self.straight_count = 0
        self.curve_cooldown = 0

        self.center_error = 999.0

        # ── [추가] 일반 차선 주행 속도 테이블 ──────────────────────────
        # 시뮬레이터 환산: code 5 == 9 km/h  ⇒  code = km/h * (5/9)
        # 목표: 최대 주행 80 km/h, 최소 주행 15 km/h
        # 기존 래더의 상대 형태(45/35/30/25/18/...)는 유지하고
        # 양 끝점만 80 / 15 km/h 로 맞춘 뒤 code 단위로 환산.
        self.KMH_TO_CODE    = 5.0 / 9.0
        self.MAX_SPEED_KMH  = 85.0
        self.MIN_SPEED_KMH  = 21.0

        self.SPD_STRAIGHT   = 12.0
        self.SPD_NEAR_STR   = 10.5
        self.SPD_NEAR       = 9.0
        self.SPD_MILD       = 8.0
        self.SPD_MOD_KEEP   = 7.0
        self.SPD_MOD_CHANGE = 6.0
        self.SPD_SHARP_TOP  = 5.5

        self.MIN_SPEED      = 5.0
        self.SHARP_K        = 0.05

        self.timer = self.create_timer(0.05, self.state_function)

        self.prev_steer=0.0
        self.state_seq = 0
        self.last_state_time = None
        self.state_dt_ms = 0.0
        self.state_hz = 0.0

        self.left_traffic=False

        self.short_lane=None

    # ── 콜백 ───────────────────────────────────────────────────────────

    def car_detect(self, msg):
        self.car = msg.data

    def overtake_active_callback(self, msg):
        """lane 코드: 추월 중(CHANGE/CHASE)이면 True"""
        self.overtake_active = bool(msg.data)

    def overtake_speed_callback(self, msg):
        """lane 코드: FSM 목표 속도 (긴급정지 반영 완료)"""
        self.overtake_speed = float(msg.data)


    def blinker_callback(self, msg):
        signal = msg.data
        self.get_logger().info(f"signal={signal}")

        if self.left_turning:
            return
        if signal in ["3_RED", "3_YELLOW", "4_RED", "4_YELLOW"]:
            self.traffic_mode = True
            self.signal = signal
        elif signal == "3_GREEN":
            self.traffic_mode = False
            self.pp_mode = True
        elif signal == "4_GREEN":
            self.traffic_mode = False
            self.signal = "LANE"
        elif signal == "4_LEFT":
            self.traffic_mode = True
            self.signal = signal

    def child_callback(self, msg):
        self.child = msg.data

    def state_callback(self, msg):
        self.steer = msg.data

    def curve_score_callback(self, msg):
        self.curve_score = float(msg.data)

    def center_error_callback(self, msg):
        self.center_error = float(msg.data)

    def path_callback(self, msg):
        if len(msg.poses) == 0:
            return
        points = [(pose.pose.position.x, pose.pose.position.y)
                  for pose in msg.poses]
        steer = self.pp.get_steer(points)
        if steer is not None:
            self.pp_steer = steer

    #레인으로 들어온 후에 일정시간 지난 후 레인 주행
    def lane_bool_callback(self, msg):
        if msg.data and self.pp_mode and self.lane_timer is None:
            self.lane_timer = self.create_timer(0.8, self.switch_to_lane)

    def human_callback(self, msg):
        self.human_detected = msg.data

    def left_obstacles_callback(self, msg):
        self.left_obstacles = msg.data

    def left_rot_callback(self, msg):
        if msg.data and not self.left_rot_doing and not self.left_rot_used:
            self.left_rot_doing = True
            self.left_rot_start_time = self.get_clock().now()

    def traffic_detected_callback(self, msg):
        self.traffic_light_detected = msg.data

    def switch_to_lane(self):
        self.pp_mode = False
        self.lane_timer.cancel()
        self.lane_timer = None

    # ── 메인 상태 함수 ─────────────────────────────────────────────────

    def state_function(self):
        cmd = Float32MultiArray()
        #라바콘 부분 pp
        if self.pp_mode:
            #신호등 노란불,빨간불일떄 정지
            if self.traffic_mode and self.signal in ["3_RED", "3_YELLOW", "4_RED", "4_YELLOW"]:
                self.steer = 0.0
                self.speed  = 0.0
            else:
                #회전 후 직진
                if self.corner_done:
                    self.steer = 0.0
                    self.speed  = 25.0
                elif self.corner_triggered:
                    elapsed = (self.get_clock().now() - self.corner_start_time).nanoseconds / 1e9
                    #처음 받은 조향각 기준으로 일정시간 회전
                    if elapsed < self.CORNER_HOLD_TIME:
                        self.steer = float(self.corner_steer)
                        abs_steer = np.abs(self.corner_steer)
                        speed = 25.0 - abs_steer * 0.4
                        self.speed = float(np.clip(speed, 18.0, 25.0))
                    else:
                        self.corner_done = True
                        self.steer = 0.0
                        self.speed  = 25.0
                #일정 조향각 이하는 무시하고 직진 후 일정 조향각 받을시 회전 진행
                else:
                    steer = self.pp_steer
                    if steer < -54.0:
                        steer = steer * 2.5
                    if steer > 53.0:
                        steer = steer * 0.85
                    if abs(steer) < 130.0:
                        steer = 0.0
                    if steer < -130.0:
                        self.corner_triggered  = True
                        self.corner_steer      = steer
                        self.corner_start_time = self.get_clock().now()
                        print(f"steer 값은 {steer}입니다")
                    self.steer = float(steer)
                    abs_steer = np.abs(steer)
                    speed = 25.0 - abs_steer * 0.8
                    self.speed = float(np.clip(speed, 10.0, 25.0))

        #신호등 판단
        elif self.traffic_mode:
            #빨강,노란불 일 시 정지
            if self.signal in ["3_RED", "3_YELLOW", "4_RED", "4_YELLOW"]:
                self.steer = 0.0
                self.speed  = 0.0
            #좌회전 신호일 떄 경찰차가 없다면 후진 후 좌회전 있다면 정지
            elif self.signal == "4_LEFT":
                if self.left_turn_time is None:
                    if not self.left_obstacles:
                        print("경찰차없음 좌회전")
                        self.left_turn_time = self.get_clock().now()
                        self.left_turning = True
                    else:
                        print("경찰차잇음")
                        self.steer = 0.0
                        self.speed  = 0.0
                else:
                    elapsed = (self.get_clock().now() - self.left_turn_time).nanoseconds / 1e9
                    if elapsed < 0.3:
                        self.steer = 0.0
                        self.speed = -30.0
                    elif elapsed < 0.3 + 1.2:
                        self.steer = -200.0
                        self.speed = 30.0                     
                    else:
                        print("좌회전 완료")
                        self.traffic_mode   = False
                        self.signal         = "LANE"
                        self.left_turn_time = None 
                        self.left_done = True
                        self.left_turning = False
                        done_msg = Bool()
                        done_msg.data = True
                        self.left_done_pub.publish(done_msg)
                        self.left_traffic=True
                        self.short_lane=self.get_clock().now()
            else:
                self.steer = self.steer
                self.speed  = 0.0
                

        #레인 일반 주행
        else:
            conend = Bool()
            self.cone_end = True
            conend.data   = self.cone_end
            self.car_pub.publish(conend)

            #첫 레인 들어갔을 떄 조향 보정을 위해 잠시 느린 속도로 주행
            # if self.count <= 12:
            #     self.speed  = 5.0
            #     self.steer = -15.0
            #     self.count += 1
            #지름길에서 나올 수 있을떄 일정시간 좌회전
            if self.left_rot_doing:
                print("지름길 좌회전 시작")
                elapsed = (self.get_clock().now() - self.left_rot_start_time).nanoseconds / 1e9
                if elapsed < 4.5:
                    self.steer = -49.0
                    self.speed  = 5.0
                    print(f"좌회전 중... {elapsed:.1f}초")
                else:
                    self.left_rot_doing = False
                    self.left_rot_used = True
                    self.left_rot_start_time = None
                    print("지름길 좌회전 완료")
                    self.left_traffic=False
            
            else:
                # car_lane 신호 항상 publish
                car_lane_msg      = Bool()
                car_lane_msg.data = self.car
                self.car_pub.publish(car_lane_msg)

                if self.human_detected:
                    # 사람 감지 → 최우선 정지
                    # cmd.angle = 0.0
                    # cmd.speed  = 0.0
                    self.steer  = self.steer
                    self.speed  = 12*self.KMH_TO_CODE

                elif self.overtake_active:
                    # ── [추가] 추월 중 (CHANGE / CHASE) ────────────────
                    # lane 코드가 계산한 조향(self.steer)과
                    # FSM 속도(self.overtake_speed)를 그대로 사용
                    self.steer  = float(self.steer)
                    self.speed  = float(self.overtake_speed)

                elif self.car:
                    # 추월 대상 감지됐지만 아직 FSM이 NORMAL인 구간
                    # (CHASE 진입 전 감속 구간)
                    self.steer  = float(self.steer)
                    abs_steer = np.abs(self.steer)
                    if abs_steer <= 2:
                        self.speed = 14.0
                    else:
                        speed = 10.0 - abs_steer * 0.1
                        self.speed = float(np.clip(speed, 5.0, 10.0))
                else:
                    # 기본 차선 주행
                    abs_prev_steer=np.abs(self.prev_steer-self.steer)
                    self.steer  = float(self.steer)
                    if self.child:
                        self.speed = 9.0
                    elif (not self.left_done) and (self.left_obstacles == False) and self.traffic_light_detected:
                        self.steer  = 0.0
                        self.speed = 0.0
                        print("경찰차 없을떄 초록불 감지")
                    elif self.left_traffic:
                        abs_steer = np.abs(self.steer)
                        short_time=(self.get_clock().now()-self.short_lane).nanoseconds / 1e9
                        if short_time<=0.7:
                            if abs_steer <= 10.0 and abs(self.center_error) <= 100.0:
                                self.speed = 25 * self.KMH_TO_CODE
                            else:
                                self.speed= 18 * self.KMH_TO_CODE
                        elif short_time<=4.0:
                            if abs_steer <= 5.0 :
                                self.speed=self.SPD_STRAIGHT
                            elif abs_steer <= 15.0:
                                self.speed=self.SPD_NEAR_STR
                            else:
                                target_speed = 25.0 - abs_steer * (0.3)
                                speed = target_speed
                                self.speed = float(np.clip(speed, 15.0, 25.0))
                        else:
                            speed=18*self.KMH_TO_CODE
                            self.speed=float(speed)
                    else:
                        abs_steer = np.abs(self.steer)

                        if abs_steer>=30.0:
                            self.speed=7.0
                        else:
                            self.speed=10.0
                        # abs_steer = np.abs(self.steer)
                        # # 진짜 직진 조건: curve_score 낮고 steer도 작아야 함
                        # abs_center_error = np.abs(self.center_error)

                        # # 급커브거나 큰 조향이 나오면, 잠깐 고속 금지
                        # if self.curve_score >= 30.0 or abs_steer >= 40.0 or abs_center_error>100.0:
                        #     self.curve_cooldown = 30

                        # if (self.curve_score <= 10.0
                        #         and abs_steer <= 15.0
                        #         and abs_center_error<=100.0
                        #         and abs_prev_steer<=7.0
                        #         and self.curve_cooldown == 0):
                        #     self.straight_count += 1
                        # else:
                        #     self.straight_count = 0

                        # if self.curve_cooldown > 0:
                        #     self.curve_cooldown -= 1

                        # # 속도 결정
                        # if self.straight_count >= 3 and self.curve_cooldown == 0:
                        #     # if abs_prev_steer<=7.0 and abs_center_error<=100.0:
                        #     cmd.speed = self.SPD_STRAIGHT
                        #     # else:
                        #     # cmd.speed = 22.0*self.KMH_TO_CODE

                        # #elif self.curve_score <= 15.0 and abs_steer <= 7.0 and self.curve_cooldown == 0:
                        # elif self.curve_score <= 15.0 and abs_steer <= 7.0:
                        #     # if abs_prev_steer<=7.0 and abs_center_error<=100.0:
                        #     cmd.speed = self.SPD_NEAR_STR
                        #     # else:
                        #     # cmd.speed = 22.0*self.KMH_TO_CODE

                        # #elif self.curve_score <= 18.0 and abs_steer <= 10.0 and self.curve_cooldown == 0:
                        # elif self.curve_score <= 18.0 and abs_steer <= 10.0:
                        #     # if abs_prev_steer<=7.0 and abs_center_error<=100.0:
                        #     cmd.speed = self.SPD_NEAR 
                        #     # else:
                        #     #     cmd.speed = 22.0*self.KMH_TO_CODE
                        #         # target_speed = self.SPD_SHARP_TOP - abs_steer * self.SHARP_K
                        #         # speed = target_speed
                        #         # cmd.speed = float(np.clip(speed, self.MIN_SPEED, self.SPD_SHARP_TOP))

                        # else:
                        #     target_speed = self.SPD_SHARP_TOP - abs_steer * self.SHARP_K
                        #     speed = target_speed
                        #     cmd.speed = float(np.clip(speed, self.MIN_SPEED, self.SPD_SHARP_TOP))
                self.prev_steer=self.steer

        # cmd = Float32MultiArray()
        cmd.data = [float(self.steer), float(self.speed)]
        self.state_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    state = State()
    try:
        rclpy.spin(state)
    except KeyboardInterrupt:
        pass
    finally:
        state.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()