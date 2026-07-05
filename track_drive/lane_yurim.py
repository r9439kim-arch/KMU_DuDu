import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
from xycar_msgs.msg import XycarMotor


left_fit_avg = None
right_fit_avg = None


# ===========================
# 추월 상태기계(FSM)
# ===========================
class OvertakeFSM:
    # 추월 상태 정의
    NORMAL, CHASE, CHANGE = 'NORMAL', 'CHASE', 'CHANGE'

    # FSM 초기 설정
    def __init__(self):
        self.FRONT_X        = 2.0
        self.TRIG_CY_MIN    = 0.0
        self.TRIG_CY_MAX    = 4.5
        self.cutin_ready_cnt = 0
        self.CUTIN_READY_HOLD = 5

        self.CUTIN_ON_LOST  = 18
        self.PASS_CY        = 0.3
        self.SIDE_BESIDE_CY = 0.55

        self.RAM_X          = 0.4
        self.DODGE_CX       = 0.5
        self.PASS_SIDE_X    = 0.5
        self.PASS_SIDE_MAX_CY = 1.0
        self.PASS_SIDE_FAR_X  = 1.2
        self.BRAKE_X        = 1.1
        self.SAFE_STOP_CY   = 1.2
        self.SAFE_CAP_SPD   = 12.0

        self.CHANGE_TRIG_CY = 0.0

        self.CHANGE_DONE_CY = 10.5
        self.SPEED_CHANGE   = 9.0
        self.SPEED_CUTIN    = 19.5
        self.CHANGE_MIN_FRAMES = 3

        self.CHANGE_MIN_FRAMES_CUTIN = 14
        self.CHANGE_MAX_FRAMES = 6
        self.CHANGE_MAX_FRAMES_CUTIN = 17 

        self.CHANGE_BLOCK_EXTRA = 15
        self.DONE_HOLD         = 3
        self.COOLDOWN          = 40
        self.COOLDOWN_CUTIN    = 2000

        self.NO_YELLOW_HOLD    = 5
        self.phase        = self.NORMAL
        self.dir          = None
        self.accelerate   = False
        self.yside_start  = None
        self.frames       = 0
        self.done_cnt     = 0
        self.cooldown     = 0
        self.lost_cnt     = 0
        self.chase_yside  = None
        self.last_cy_seen = 99.0
        self.passed_side  = None
        self.no_yellow_cnt = 0

    @staticmethod
    def opp(s):
        return 'right' if s == 'left' else 'left'

    # 추월 상태 초기화
    def force_normal(self):
        self.phase           = self.NORMAL
        self.dir             = None
        self.accelerate      = False
        self.yside_start     = None
        self.frames          = 0
        self.done_cnt        = 0
        self.lost_cnt        = 0
        self.cutin_ready_cnt = 0
        self.chase_yside     = None
        self.no_yellow_cnt   = 0
        self.cooldown        = self.COOLDOWN   # 바로 재추월 방지

    # 앞차와의 거리 기반 속도 제한
    def front_speed_cap(self, detected, cx, cy):
        if not (detected and abs(cx) < self.BRAKE_X):
            return None
        if cy <= self.SAFE_STOP_CY:
            return 0.0
        if cy >= self.SAFE_BRAKE_CY:
            return None
        ratio = (cy - self.SAFE_STOP_CY) / (self.SAFE_BRAKE_CY - self.SAFE_STOP_CY)
        return ratio * self.SAFE_CAP_SPD

    # 차선 변경 방향 결정
    def _change_dir(self, cx, yside):
        if abs(cx) >= self.DODGE_CX:
            return 'right' if cx < 0 else 'left'
        if yside in ('left', 'right'):
            return yside
        return 'right' if cx < 0 else 'left'

    # 차선 변경 시작
    def _start_change(self, dir_side, accel, yellow_side=None):
        self.dir         = dir_side
        self.yside_start = yellow_side if yellow_side in ('left', 'right') else dir_side
        self.accelerate  = accel
        self.phase       = self.CHANGE
        self.frames      = 0
        self.done_cnt    = 0
        self.cutin_ready_cnt = 0
        self.no_yellow_cnt   = 0

    # Cut-in 진입 조건 확인
    def _cutin_hold_ready(self, cx, cy, side, cy_thresh=None, hold=None):
        thresh   = cy_thresh if cy_thresh is not None else self.SIDE_BESIDE_CY
        required = hold      if hold      is not None else self.CUTIN_READY_HOLD
        cutin_ready = (
            cy <= thresh and
            abs(cx) > self.RAM_X and
            side in ('left', 'right')
        )
        if cutin_ready:
            self.cutin_ready_cnt += 1
        else:
            self.cutin_ready_cnt = 0
        return self.cutin_ready_cnt >= required

    # 추월 시작 조건 확인
    def _try_trigger(self, cx, cy, yside):
        if abs(cx) < self.FRONT_X:
            self.cutin_ready_cnt = 0
            if cy >= self.CHANGE_TRIG_CY:

                side = self._change_dir(cx, yside)
                self._start_change(side, accel=False, yellow_side=yside)
            return

        side = yside if yside in ('left', 'right') else ('left' if cx < 0 else 'right')
        if self._cutin_hold_ready(cx, cy, side):
            self._start_change(side, accel=True)
        else:
            self.phase = self.CHASE
            self.lost_cnt = 0
            self.chase_yside = side
            self.last_cy_seen = cy

    # FSM 상태 갱신
    def update(self, detected, cx, cy, yside):
        if self.cooldown > 0:
            self.cooldown -= 1

        if self.phase == self.NORMAL:
            if detected and self.TRIG_CY_MIN < cy < self.TRIG_CY_MAX:
                if self.cooldown > 0:
                    near_side = 'right' if cx > 0 else 'left'
                    if (near_side == self.passed_side and cy < 1.5
                            and abs(cx) < self.FRONT_X):
                        return self.decision()
                self._try_trigger(cx, cy, yside)

        elif self.phase == self.CHASE:
            if not detected:
                self.cutin_ready_cnt = 0
                self.lost_cnt += 1
                if self.lost_cnt >= self.CUTIN_ON_LOST:
                    if self.chase_yside in ('left', 'right') and self.last_cy_seen < self.PASS_CY:
                        self._start_change(self.chase_yside, accel=True)
                    else:
                        self.phase = self.NORMAL
            else:
                self.lost_cnt = 0
                self.last_cy_seen = cy
                self.chase_yside = 'left' if cx < 0 else 'right'

                if self._cutin_hold_ready(cx, cy, self.chase_yside, cy_thresh=1.2, hold=5): # 여기
                    self._start_change(self.chase_yside, accel=True)
                elif abs(cx) < self.FRONT_X \
                        and self.CHANGE_TRIG_CY <= cy < self.TRIG_CY_MAX:
                    side = self._change_dir(cx, yside)
                    self.cutin_ready_cnt = 0
                    self._start_change(side, accel=False, yellow_side=yside)
                elif cy >= self.TRIG_CY_MAX:
                    self.cutin_ready_cnt = 0
                    self.phase = self.NORMAL

        elif self.phase == self.CHANGE:
            self.frames += 1
            crossed = (yside is not None and yside == self.opp(self.yside_start))
            if crossed:
                self.done_cnt += 1
            elif yside is None:
                pass
            else:
                # 반대 방향(아직 안 넘음)으로 확실히 잡힐 때만 리셋
                self.done_cnt = 0

            if yside is None:
                self.no_yellow_cnt += 1
            else:
                self.no_yellow_cnt = 0

            min_frames = (self.CHANGE_MIN_FRAMES_CUTIN if self.accelerate
                          else self.CHANGE_MIN_FRAMES)
            can_finish = self.frames >= min_frames
            no_yellow_limit = min_frames + 5

            forced_no_yellow = (
                self.no_yellow_cnt >= self.NO_YELLOW_HOLD and
                self.frames >= no_yellow_limit
            )
            if forced_no_yellow:
                can_finish = True
                self.done_cnt = self.DONE_HOLD

            # ── 앞차가 아직 정면(±RAM_X)에 가까이 있으면 종료 금지 ──
            #    차량이 옆으로 빠졌거나(|cx|>=RAM_X) / 안 보이거나 / 뒤로 갔을 때만 종료 허용
            still_blocking = (detected and abs(cx) < self.RAM_X
                              and self.SIDE_BESIDE_CY < cy < self.CHANGE_DONE_CY)

            print(f'[CHANGE] f={self.frames} yside={yside} crossed={crossed} '
                  f'done={self.done_cnt} noY={self.no_yellow_cnt} '
                  f'cx={cx:.2f} cy={cy:.2f} block={still_blocking}')

            normal_finish = (can_finish and self.done_cnt >= self.DONE_HOLD) and not still_blocking
            # ── 앞차를 이미 충분히 추월(멀리 앞섬)했으면 최소 프레임만 채우고 조기 종료 ──
            #    노란선 크로스(done_cnt)를 못 봤더라도 위험이 사라졌으므로 핸들을 푼다.
            passed_far = (detected and cy >= self.CHANGE_DONE_CY and can_finish)
            passed_aside = (
                detected and can_finish and (
                    (self.dir == 'right' and cx <= -self.PASS_SIDE_X) or
                    (self.dir == 'left'  and cx >=  self.PASS_SIDE_X)
                )
                # ── 정면 변경: 가깝거나(cy 작음) 옆으로 확실히 빠졌으면(|cx| 큼) 종료.
                #    cut-in 은 예외. (멀리 앞-좌 cx=-1.5,cy=4 같은 건 계속 회피)
                and (self.accelerate
                     or cy < self.PASS_SIDE_MAX_CY
                     or abs(cx) >= self.PASS_SIDE_FAR_X)
            )
            # cut-in(accel=True)은 전용 상한, 정면 변경은 기본 상한 사용
            max_frames = (self.CHANGE_MAX_FRAMES_CUTIN if self.accelerate
                          else self.CHANGE_MAX_FRAMES)

            # ── [A] 앞차가 아직 막고 있으면 max_frames 를 넘겨도 풀지 않고 계속 회피.
            #    단 abs_max(절대 한도)까지만 — 그 뒤엔 안전상 무조건 종료.
            abs_max = max_frames + self.CHANGE_BLOCK_EXTRA
            if self.frames > abs_max:
                hard_finish = True            # 절대 한도: 무조건 종료
            elif still_blocking:
                hard_finish = False           # 아직 앞차가 막으면 풀지 말고 계속 빼냄
            else:
                hard_finish = self.frames > max_frames

            if normal_finish or hard_finish or passed_far or passed_aside:
                reason = ('MAX_FRAMES' if hard_finish
                          else 'PASSED' if passed_far
                          else 'PASSED_SIDE' if passed_aside
                          else ('NO_YELLOW' if forced_no_yellow else 'CROSSED'))
                print(f'[CHANGE END] reason={reason} frames={self.frames} cy={cy:.2f}')
                self.passed_side = self.opp(self.dir)
                self.phase    = self.NORMAL
                self.dir      = None
                # cut-in(accel=True)은 전용 쿨다운, 정면 변경은 기본 쿨다운
                self.cooldown = (self.COOLDOWN_CUTIN if self.accelerate
                                 else self.COOLDOWN)

        return self.decision()

    # 현재 FSM 결과 반환
    def decision(self):
        if self.phase == self.CHANGE:
            spd = self.SPEED_CUTIN if self.accelerate else self.SPEED_CHANGE
            return dict(phase=self.phase, change=True, chase=False, dir=self.dir,
                        speed=spd, accelerate=self.accelerate)
        if self.phase == self.CHASE:
            return dict(phase=self.phase, change=False, chase=True, dir=None,
                        speed=self.SPEED_CUTIN, accelerate=False)
        return dict(phase=self.NORMAL, change=False, chase=False, dir=None,
                    speed=None, accelerate=False)


# ===========================
# 차선 검출 노드
# ===========================
class LaneDetectionNode(Node):

    # ── CHANGE 시 고정 조향각 ──────────────────────────────────────────
    CHANGE_STEER = 29.0

    # 노드 및 ROS 통신 초기화
    def __init__(self):
        super().__init__('lane_detection_node')

        # ── Subscriber ─────────────────────────────────────────────────
        self.cam_sub = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, 10)

        self.mid_sub = self.create_subscription(
            Point, '/mid_point_xy', self.mid_callback, qos_profile_sensor_data)

        self.det_sub = self.create_subscription(
            Bool, '/large_vehicle_detected', self.detection_callback, 10)

        self.car_l = self.create_subscription(
            Bool, '/car_lane', self.car_lane_callback, 10)

        self.motor_sub = self.create_subscription(
            XycarMotor, '/xycar_motor', self.motor_callback, 10)

        # ── Publisher ──────────────────────────────────────────────────
        # 일반 주행용
        self.pub_bool  = self.create_publisher(Bool,    '/lane_bool',  10)
        self.pub_steer = self.create_publisher(Float32, '/lane_steer', 10)
        self.pub_child = self.create_publisher(Bool,    '/lane_child', 10)
        self.pub_curve_score = self.create_publisher(Float32, '/lane_curve_score', 10)
        self.pub_center_error = self.create_publisher(Float32, '/lane_center_error', 10)

        # 추월 상태 → state.py 에서 속도·모터 최종 결정에 사용
        self.pub_overtake_active = self.create_publisher(Bool,    '/overtake_active', 10)
        self.pub_overtake_speed  = self.create_publisher(Float32, '/overtake_speed',  10)

        self.bridge = CvBridge()

        self.src_points = np.float32([
            [145, 295],
            [10,  370],
            [485, 295],
            [630, 370]
        ])
        self.bev_width  = 640
        self.bev_height = 480
        self.dst_points = np.float32([
            [0,              0             ],
            [0,              self.bev_height],
            [self.bev_width, 0             ],
            [self.bev_width, self.bev_height]
        ])

        self.n_windows     = 12
        self.margin        = 50
        self.minpix        = 20
        self.window_height = self.bev_height // self.n_windows
        self.target_indxs  = [2, 3, 5, 8, 9]
        self.steer         = 0.0
        self.lane_state    = True
        self.lane_width    = 290.0
        self.target        = 320.0

        self.left_centers_count  = 0
        self.right_centers_count = 0
        self.prev_steer = 0.0
        self.pre_steer  = 0.0
        self.alpha      = 0.4
        self.max_step   = 6.0

        # 환산: code = km/h * 5/9 (시뮬 code 5 == 9 km/h)
        # 고속일수록 max_step·alpha 를 낮춰 조향을 더 부드럽게(진동 억제).
        # 곡률→조향 크기 게인 테이블은 건드리지 않음.
        self.KMH_TO_CODE   = 5.0 / 9.0
        self.cmd_speed     = 0.0          # state.py /xycar_motor 지령 속도(code)
        self.DAMP_SPD_LO   = 45.0 * self.KMH_TO_CODE   # 이하: 댐핑 없음 (커브 영역 보호)
        self.DAMP_SPD_HI   = 82.0 * self.KMH_TO_CODE   # 이상: 최대 댐핑 (직선 전용)
        self.MAXSTEP_DAMP  = 0.15         # 고속에서 max_step 최대 15% 감소
        self.ALPHA_DAMP    = 0.12         # 고속에서 alpha 최대 12% 감소
        self.STEER_DEG_PER_CODE = 0.2     # 참고: code 100 == 20deg

        self.mode      = ''
        self.lost_count = 0
        self.mode_cand  = ''
        self.mode_cand_n = 0
        self.child      = False

        self.curve_score = 99.0
        self.center_error = 999.0 

        self.result_w = None
        self.result_y = None
        self.img_bev_w = None
        self.img_bev_y = None
        self.binary_w  = None
        self.binary_y  = None

        self.left_start_color  = None
        self.right_start_color = None
        self.right_main_color  = None
        self.left_main_color   = None

        # ── 추월 관련 상태 ─────────────────────────────────────────────
        self.fsm     = OvertakeFSM()
        self.fsm_dec = self.fsm.decision()   # 초기값 NORMAL

        # ── cut-in 종료 후 차선-주행 잠금 ─────────────────────────────
        #    cut-in(accel=True) 추월이 끝나면 이 시간 동안은 무조건 NORMAL(차선 주행)
        #    만 하고, 재추월/CHASE 로 넘어가지 않는다. (어보구가 들어와도 어차피
        #    NORMAL 안의 school_zone 모드라 그대로 차선 주행이 된다.)
        self.LANE_LOCK_SEC   = 5.0
        self.lane_lock_start = None          # 잠금 시작 시각(rclpy Time), None=비활성
        self._prev_phase     = OvertakeFSM.NORMAL
        self._prev_accel     = False

        # 라이다 수신 데이터
        self.detected  = False
        self.last_cx   = 0.0
        self.last_cy   = 99.0
        self.det_timeout = 4.0

        # ── 좌우 폭 게이트 (비대칭) ───────────────────────────────────────
        # 라이다 x(좌우 치우침)가 이 값보다 크면 옆 차로/먼 쪽 차량으로 보고
        # 추월 FSM 에서 무시한다. cx<0=좌측, cx>0=우측.
        #   좌측: SIDE_LIMIT_X(넓게)  /  우측: SIDE_LIMIT_X_RIGHT(좁게, +1m)
        self.SIDE_LIMIT_X       = 3.0   # 좌측 한계
        self.SIDE_LIMIT_X_RIGHT = 3.0   # 우측 한계

        now = self.get_clock().now()
        self.last_bool_stamp = now
        self.last_mid_stamp  = now
        self.last_det_stamp  = now
        self._last_trig      = None

        self.car_det = False

        # yellow_side 디버그
        self.dbg_yellow_cx = -1.0
        self.dbg_yellow_n  = 0

        # yellow_side 파라미터
        self.YELLOW_MIN_PIX    = 150
        self.YELLOW_MARGIN     = 30
        self.yellow_open_flip  = False
        self.YELLOW_MAX_LINE_W = 80
        self.YELLOW_MIN_BLOB   = 40

        # BEV 변환 행렬
        src1 = np.float32([[202,286],[427,284],[635,435],[2,433]])
        dst1 = np.float32([
            [0,              0             ],
            [self.bev_width, 0             ],
            [self.bev_width, self.bev_height],
            [0,              self.bev_height]
        ])
        self.M_yside = cv2.getPerspectiveTransform(src1, dst1)

        self.get_logger().info('LaneDetectionNode (통합) 시작')

    # ── 콜백 ───────────────────────────────────────────────────────────
    def detection_callback(self, msg):
        self.detected = bool(msg.data)
        now = self.get_clock().now()
        self.last_bool_stamp = now
        self.last_det_stamp  = now
        if self._last_trig != msg.data:
            self.get_logger().info(f'[detect] large_vehicle={msg.data}')
            self._last_trig = msg.data

    def car_lane_callback(self, msg):
        self.car_det = bool(msg.data)

    def motor_callback(self, msg):
        self.cmd_speed = float(msg.speed)

    def _speed_damp_frac(self):
        lo, hi = self.DAMP_SPD_LO, self.DAMP_SPD_HI
        if hi <= lo:
            return 0.0
        return float(np.clip((abs(self.cmd_speed) - lo) / (hi - lo), 0.0, 1.0))

    def mid_callback(self, msg):
        self.last_cx = float(msg.x)
        self.last_cy = float(msg.y)
        now = self.get_clock().now()
        self.last_mid_stamp = now
        self.last_det_stamp = now

    # ── yellow_side (코드1 BEVLaneDetector.yellow_side 이식) ───────────
    def yellow_side(self, yellow_mask):
        """노란선이 BEV 기준 좌/우 어느 쪽인지 반환"""
        yb = cv2.warpPerspective(yellow_mask, self.M_yside,
                                 (self.bev_width, self.bev_height))
        kernel = np.ones((5, 5), np.uint8)
        yb = cv2.morphologyEx(yb, cv2.MORPH_CLOSE, kernel)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(yb, connectivity=8)
        line_mask = np.zeros_like(yb)
        for i in range(1, num):
            w = stats[i, cv2.CC_STAT_WIDTH]
            a = stats[i, cv2.CC_STAT_AREA]
            if w <= self.YELLOW_MAX_LINE_W and a >= self.YELLOW_MIN_BLOB:
                line_mask[labels == i] = 255
        yb = line_mask

        ys, xs = yb.nonzero()
        self.dbg_yellow_n = int(len(xs))
        if len(xs) < self.YELLOW_MIN_PIX:
            self.dbg_yellow_cx = -1.0
            return None

        cx = float(np.mean(xs))
        self.dbg_yellow_cx = cx
        center = self.bev_width / 2.0

        if abs(cx - center) < self.YELLOW_MARGIN:
            return None

        side = 'left' if cx < center else 'right'
        if self.yellow_open_flip:
            side = 'right' if side == 'left' else 'left'
        return side

    # ── 카메라 콜백 ────────────────────────────────────────────────────
    def cam_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0,   0,   220])
        upper_white = np.array([180, 30,  255])
        mask_white  = cv2.inRange(img_hsv, lower_white, upper_white)

        lower_yellow = np.array([18, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow  = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        mask_white  = cv2.GaussianBlur(mask_white,  (5, 5), 0)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)

        _, self.result_w = cv2.threshold(mask_white,  127, 255, cv2.THRESH_BINARY)
        _, self.result_y = cv2.threshold(mask_yellow, 127, 255, cv2.THRESH_BINARY)

        # ── FSM 갱신 ──────────────────────────────────────────────────
        now = self.get_clock().now()
        bool_age = (now - self.last_bool_stamp).nanoseconds * 1e-9
        mid_age  = (now - self.last_mid_stamp ).nanoseconds * 1e-9

        # 좌우 폭 게이트: 라이다 위치가 신선한지 / 좌우 1.5m 이내인지
        mid_fresh     = (mid_age < self.det_timeout)
        # 비대칭: 우측(cx>0)은 SIDE_LIMIT_X_RIGHT, 좌측(cx<0)은 SIDE_LIMIT_X
        side_limit = (self.SIDE_LIMIT_X_RIGHT if self.last_cx > 0
                      else self.SIDE_LIMIT_X)
        within_lateral = (abs(self.last_cx) <= side_limit)

        raw_recent = self.detected and (bool_age < self.det_timeout)
        mid_valid  = (
            mid_fresh and
            self.fsm.TRIG_CY_MIN < self.last_cy < self.fsm.TRIG_CY_MAX and
            within_lateral
        )
        fsm_detected = raw_recent or mid_valid or self.car_det

        # 라이다 점이 신선한데 좌우로 1.5m 밖이면 옆 차로 차량 → 추월 대상에서 제외
        if mid_fresh and not within_lateral:
            fsm_detected = False

        yside = self.yellow_side(self.result_y)

        # ── cut-in 종료 후 차선-주행 잠금 처리 ────────────────────────
        #    잠금 중이면 FSM 을 돌리지 않고 강제로 NORMAL 유지 → 추월 트리거 차단.
        lane_lock_active = (
            self.lane_lock_start is not None and
            (now - self.lane_lock_start).nanoseconds * 1e-9 < self.LANE_LOCK_SEC
        )

        if lane_lock_active:
            # 추월 상태기계를 멈추고 무조건 차선 주행만 한다.
            self.fsm.phase = OvertakeFSM.NORMAL
            self.fsm.dir   = None
            self.fsm_dec   = self.fsm.decision()
        else:
            self.lane_lock_start = None   # 만료된 잠금 정리
            self.fsm_dec = self.fsm.update(
                detected=fsm_detected,
                cx=self.last_cx,
                cy=self.last_cy,
                yside=yside
            )
            # cut-in(accel=True) → NORMAL 전환을 감지하면 잠금 시작
            if (self._prev_phase == OvertakeFSM.CHANGE and self._prev_accel
                    and self.fsm_dec['phase'] == OvertakeFSM.NORMAL):
                self.lane_lock_start = now
                self.get_logger().info(
                    f'[LANE_LOCK] cut-in 종료 → {self.LANE_LOCK_SEC:.0f}s 차선 주행 고정')

        # 다음 프레임 비교용 직전 상태 갱신
        self._prev_phase = self.fsm_dec['phase']
        self._prev_accel = self.fsm.accelerate

        # ── 긴급 정지 속도 상한 계산 ──────────────────────────────────
        # 긴급 속도 상한: state.py 의 overtake_speed 에 반영하기 위해 계산
        self._emergency_speed_cap = None

        # ── [정지 게이트] 근거리 정지는 large_vehicle 이 최근에 True 일 때만 ──
        #    raw_recent = self.detected(/large_vehicle_detected) 이 det_timeout 이내.
        #    mid_point(라이다 점)나 /car_lane 만으로는 정지하지 않게 한다.
        if self.fsm_dec['phase'] in (OvertakeFSM.NORMAL, OvertakeFSM.CHASE):
            cap = self.fsm.front_speed_cap(raw_recent, self.last_cx, self.last_cy)
            if cap is not None:
                self._emergency_speed_cap = cap

        if mid_age < 1.0 and self.last_cy <= self.fsm.SAFE_STOP_CY:
            cx_abs = abs(self.last_cx)
            if self.fsm_dec['phase'] == OvertakeFSM.CHANGE:
                if cx_abs < 0.9:                      # 차폭 겹침 = 박기 직전
                    self._emergency_speed_cap = 4.0   # 저속(회피 지속), 정지는 안 함
                # else: 옆으로 빠짐 → cap 없음
            elif cx_abs < self.fsm.BRAKE_X and raw_recent:
                self._emergency_speed_cap = 0.0

        self.birdeyeview()

    def build_yellow_virtual_line(self, binary_lane_y):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_lane_y, connectivity=8
        )

        points = []

        for i in range(1, num):
            x, y, w, h, area = stats[i]

            # 노이즈 제거
            if area < 25:
                continue

            # 너무 큰 노란 영역 제거
            if area > 2500:
                continue

            # 노란 점선은 너무 넓으면 안 됨
            if w > 90:
                continue

            cx, cy = centroids[i]
            points.append((cx, cy))

        if len(points) < 3:
            return binary_lane_y

        points = np.array(points)
        ys = points[:, 1]
        xs = points[:, 0]

        fit = np.polyfit(ys, xs, 2)

        virtual = np.zeros_like(binary_lane_y)

        y_vals = np.linspace(0, self.bev_height - 1, 80)
        x_vals = fit[0] * y_vals**2 + fit[1] * y_vals + fit[2]

        pts = []
        for x, y in zip(x_vals, y_vals):
            if 0 <= x < self.bev_width:
                pts.append([int(x), int(y)])

        if len(pts) >= 2:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(virtual, [pts], False, 255, thickness=5)

        return cv2.bitwise_or(binary_lane_y, virtual)

    # ── BEV + 슬라이딩 윈도우 ─────────────────────────────────────────
    def birdeyeview(self):
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        L_bev_w = cv2.warpPerspective(self.result_w, M,
                                      (self.bev_width, self.bev_height))
        _, binary_lane_w = cv2.threshold(L_bev_w, 120, 255, cv2.THRESH_BINARY)

        L_bev_y = cv2.warpPerspective(self.result_y, M,
                                      (self.bev_width, self.bev_height))
        _, binary_lane_y = cv2.threshold(L_bev_y, 120, 255, cv2.THRESH_BINARY)

        binary_lane_y = self.remove_road_text(binary_lane_y)

        # 노란 점선 가상선 보강
        binary_lane_y = self.build_yellow_virtual_line(binary_lane_y)

        self.img_bev_w = L_bev_w
        self.img_bev_y = L_bev_y
        self.binary_w  = binary_lane_w
        self.binary_y  = binary_lane_y
        self.binary    = cv2.bitwise_or(self.binary_w, self.binary_y)

        self.histogram(self.binary)

    # 차선 시작 위치 탐색
    def histogram(self, pro_img):
        height = pro_img.shape[0]
        partial_total = pro_img[height * 1 // 3:, :]

        hist_total = np.sum(partial_total, axis=0)
        self.left_start  = int(np.argmax(hist_total[:self.bev_width // 2])) - 10 \
            if np.any(hist_total[:self.bev_width // 2] > 0) else 1
        self.right_start = int(np.argmax(hist_total[self.bev_width // 2:]) + self.bev_width // 2) + 10 \
            if np.any(hist_total[self.bev_width // 2:] > 0) else self.bev_width - 1

        self.left_start_color  = self.get_lane_color(self.left_start,
                                                      height * 3 // 4, height)
        self.right_start_color = self.get_lane_color(self.right_start,
                                                      height * 3 // 4, height)
        self.sliding_window(self.left_start, self.right_start,
                            self.left_start_color, self.right_start_color)

    # 차선 색상 판별
    def get_lane_color(self, x, y_low, y_high):
        margin = 30
        x1 = max(0, x - margin)
        x2 = min(self.bev_width, x + margin)
        roi_w = self.binary_w[y_low:y_high, x1:x2]
        roi_y = self.binary_y[y_low:y_high, x1:x2]
        white_count  = cv2.countNonZero(roi_w)
        yellow_count = cv2.countNonZero(roi_y)
        if yellow_count > white_count and yellow_count > 30:
            return 'yellow'
        elif white_count > yellow_count and white_count > 50:
            return 'white'
        return 'unknown'

    # 노면 글자 제거
    def remove_road_text(self, mask):
        if mask is None or cv2.countNonZero(mask) == 0:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 5))
        merged = cv2.dilate(mask, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        text_region = np.zeros_like(mask)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            aspect = w / float(h) if h > 0 else 99.0
            if w >= 130 and h <= 120 and aspect >= 1.4:
                text_region[labels == i] = 255
        return cv2.bitwise_and(mask, cv2.bitwise_not(text_region))

    # 직선 차선 이상점 제거
    def filter_straight_outliers(self, x_coords, y_coords,
                                 slope_band=0.15, min_points=3):
        xs = np.asarray(x_coords, dtype=float)
        ys = np.asarray(y_coords, dtype=float)
        n  = len(xs)
        if n < min_points + 1:
            return x_coords, y_coords
        order = np.argsort(ys)
        xs, ys = xs[order], ys[order]
        dy = np.diff(ys)
        dx = np.diff(xs)
        seg_slope = np.divide(dx, dy, out=np.zeros_like(dx), where=dy != 0)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            slopes = []
            if i > 0:     slopes.append(abs(seg_slope[i - 1]))
            if i < n - 1: slopes.append(abs(seg_slope[i]))
            if slopes and min(slopes) > slope_band:
                keep[i] = False
        if keep.sum() < min_points:
            return x_coords, y_coords
        return xs[keep].tolist(), ys[keep].tolist()

    # 슬라이딩 윈도우 차선 추적
    def sliding_window(self, left_start, right_start, left_color, right_color):
        self.right_main_color = right_color
        self.left_main_color  = left_color
        self.left_missing_count  = 0
        self.right_missing_count = 0

        window_height = self.bev_height // self.n_windows
        nonzero  = self.binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        curr_left_x  = left_start
        curr_right_x = right_start
        left_lane_inds  = []
        right_lane_inds = []
        self.left_centers  = []
        self.right_centers = []

        debug_img = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2BGR)

        for window in range(self.n_windows):
            win_y_low  = self.bev_height - (window + 1) * window_height
            win_y_high = self.bev_height - window * window_height

            win_xleft_left   = curr_left_x  - self.margin
            win_xleft_right  = curr_left_x  + self.margin
            win_xright_left  = curr_right_x - self.margin
            win_xright_right = curr_right_x + self.margin

            cv2.rectangle(debug_img,
                          (win_xleft_left,  win_y_low),
                          (win_xleft_right, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(debug_img,
                          (win_xright_left,  win_y_low),
                          (win_xright_right, win_y_high), (0, 255, 0), 2)

            good_left_indx  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xleft_left) &
                               (nonzerox < win_xleft_right)).nonzero()[0]
            good_right_indx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_left) &
                               (nonzerox < win_xright_right)).nonzero()[0]

            left_lane_inds.append(good_left_indx)
            right_lane_inds.append(good_right_indx)

            if len(good_left_indx) > self.minpix:
                curr_left_x = int(np.mean(nonzerox[good_left_indx]))
            else:
                self.left_missing_count += 1
                curr_left_x = self.left_centers[-1] \
                    if len(self.left_centers) > 0 else curr_left_x

            if len(good_right_indx) > self.minpix:
                curr_right_x = int(np.mean(nonzerox[good_right_indx]))
            else:
                self.right_missing_count += 1
                curr_right_x = self.right_centers[-1] \
                    if len(self.right_centers) > 0 else curr_right_x

            self.left_centers.append(curr_left_x)
            self.right_centers.append(curr_right_x)
            self.child = False

        left_mean  = np.mean(self.left_centers)
        right_mean = np.mean(self.right_centers)
        lane_gap   = abs(left_mean - right_mean)

        cv2.waitKey(1)

        # ── 추월 FSM 우선 분기 ────────────────────────────────────────
        dec = self.fsm_dec

        if dec['phase'] == OvertakeFSM.CHANGE:
            # CHANGE: 고정 조향각을 /lane_steer 로 publish
            # 속도는 /overtake_speed 로 publish → state.py 가 최종 모터 명령
            steer_sign  = 1.0 if dec['dir'] == 'right' else -1.0
            change_steer = steer_sign * self.CHANGE_STEER

            # 긴급 속도 상한 반영
            speed = dec['speed']
            if self._emergency_speed_cap is not None:
                speed = min(speed, self._emergency_speed_cap)

            self._publish_overtake(change_steer, speed, active=True)
            self.lane_state = True
            self.publish_bool()
            return

        if dec['phase'] == OvertakeFSM.CHASE:
            # CHASE: 차선 추종 조향값 계산 후 /lane_steer 로 publish
            self._calc_lane_steer_for_chase(left_color, right_color, lane_gap)

            speed = dec['speed']
            if self._emergency_speed_cap is not None:
                speed = min(speed, self._emergency_speed_cap)

            self._publish_overtake(self.steer, speed, active=True)
            self.lane_state = True
            self.publish_bool()
            return

        # ── NORMAL 모드 분기 ─────────────────────────────
        raw_mode = ''
        if left_color == 'yellow' and right_color == 'white':
            raw_mode = 'straight'
        elif left_color == 'white' and right_color == 'yellow':
            raw_mode = 'straight2'
        elif left_color == 'white' and right_color == 'white':
            if lane_gap < 150:
                raw_mode = 'only_right' if self.left_start > self.target else 'only_left'
            else:
                raw_mode = 'curve'
        elif left_color == 'yellow' and right_color == 'yellow':
            raw_mode = 'school_zone'
            self.child = True
        elif left_color == 'unknown' and right_color == 'yellow':
            raw_mode = 'school_zone'
            self.child = True
        elif left_color == 'yellow' and right_color == 'unknown':
            raw_mode = 'school_zone'
            self.child = True
        else:
            raw_mode = 'straight'

        # 모드 히스테리시스
        if raw_mode == self.mode_cand:
            self.mode_cand_n += 1
        else:
            self.mode_cand   = raw_mode
            self.mode_cand_n = 1
        if self.mode_cand_n >= 3:
            self.mode = raw_mode

        if self._emergency_speed_cap is not None:
            active_msg = Bool()
            active_msg.data = True
            self.pub_overtake_active.publish(active_msg)

            speed_msg = Float32()
            speed_msg.data = float(np.clip(self._emergency_speed_cap, 0.0, 30.0))
            self.pub_overtake_speed.publish(speed_msg)

            self.get_logger().info(
                f'[EMERGENCY] NORMAL 근거리 앞차 → speed_cap={self._emergency_speed_cap:.1f} '
                f'cx={self.last_cx:.2f} cy={self.last_cy:.2f}')
        else:
            # NORMAL 상태: overtake 비활성 신호 송출
            self._publish_overtake_inactive()

        self.update_curve_score()
        self.publish_curve_score()

        self.publish_center_error()
        self.calculate_steer()

    # ── 추월 중 조향 계산 (CHASE 전용) ────────────────────────────────
    def _calc_lane_steer_for_chase(self, left_color, right_color, lane_gap):
        left_missing  = self.left_missing_count  > 6
        right_missing = self.right_missing_count > 6

        if left_missing and right_missing:
            self.steer = self.prev_steer
            return

        if self.mode in ('straight', 'straight2'):
            self._straight_calc()
        elif self.mode == 'curve':
            self._curve_calc()
        else:
            self._straight_calc()

    # ── 일반 조향 계산 ─────────────────────────────────────────────────
    def calculate_steer(self):
        if len(self.left_centers) < 12 or len(self.right_centers) < 12:
            return

        left_missing  = self.left_missing_count  > 6
        right_missing = self.right_missing_count > 6

        if self.right_missing_count - self.left_missing_count >= 4:
            self.only_left()
            return
        elif self.left_missing_count - self.right_missing_count >= 4:
            self.only_right()
            return
        elif left_missing and right_missing:
            print("양쪽 차선 다 안 잡힘")
            self.lost_count += 1
            self.steer = self.prev_steer if self.lost_count <= 12 \
                else self.prev_steer * 0.7
            self.lane_state = False
            self.publish_steer()
            self.publish_bool()
            return
        else:
            self.lost_count = 0

        if self.mode == 'straight':
            self.straight()
        elif self.mode == 'straight2':
            self.straight2()
        elif self.mode == 'curve':
            self.curve()
        elif self.mode == 'school_zone':
            self.school_zone()

    # ── Publish 함수들 ─────────────────────────────────────────────────

    def _publish_overtake(self, steer_val: float, speed: float, active: bool):

        steer_msg = Float32()
        steer_msg.data = float(steer_val)
        self.pub_steer.publish(steer_msg)

        active_msg = Bool()
        active_msg.data = active
        self.pub_overtake_active.publish(active_msg)

        speed_msg = Float32()
        speed_msg.data = float(np.clip(speed, 0.0, 30.0))
        self.pub_overtake_speed.publish(speed_msg)

        self.get_logger().info(
            f'[OVERTAKE] phase={self.fsm_dec["phase"]} '
            f'dir={self.fsm_dec["dir"]} '
            f'steer={steer_val:.1f} speed={speed:.1f}')

    def _publish_overtake_inactive(self):
        active_msg = Bool()
        active_msg.data = False
        self.pub_overtake_active.publish(active_msg)

    def publish_steer(self):
        if abs(self.prev_steer) > 40 and \
                np.sign(self.steer) != np.sign(self.prev_steer):
            self.max_step = min(self.max_step, 20)

        self.pre_steer = self.steer
        lpf_target   = (1.0 - self.alpha) * self.prev_steer + self.alpha * self.steer
        diff         = lpf_target - self.prev_steer
        clipped_diff = np.clip(diff, -self.max_step, self.max_step)
        final_steer  = self.prev_steer + clipped_diff
        self.prev_steer = final_steer

        steer_msg = Float32()
        steer_msg.data = float(final_steer)
        self.pub_steer.publish(steer_msg)

        child_msg = Bool()
        child_msg.data = bool(self.child)
        self.pub_child.publish(child_msg)

        print(f'[NORMAL] mode: {self.mode} | steer={final_steer:.1f}')

    def publish_bool(self):
        msg = Bool()
        msg.data = self.lane_state
        self.pub_bool.publish(msg)

    # 곡률 계산
    def update_curve_score(self):
        if len(self.left_centers) < 12 or len(self.right_centers) < 12:
            self.curve_score = 99.0
            self.center_error = 999.0
            return

        x_coords, y_coords = [], []

        for i in self.target_indxs:
            center_x = (self.left_centers[i] + self.right_centers[i]) / 2
            y = self.bev_height - (i * self.window_height) - (self.window_height / 2)

            x_coords.append(center_x)
            y_coords.append(y)

        a, b, c = np.polyfit(y_coords, x_coords, 2)

        self.curve_score = float(abs(a) * 100000.0)
        self.center_error = float(np.mean(x_coords) - self.target)

    def publish_curve_score(self):
        msg = Float32()
        msg.data = float(self.curve_score)
        self.pub_curve_score.publish(msg)

    def publish_center_error(self):
        msg = Float32()
        print(f"center_error: {self.center_error}")
        msg.data = float(self.center_error)
        if self.mode=='curve':
            msg.data=0.0
            self.pub_center_error.publish(msg)
        else:
            self.pub_center_error.publish(msg)

    # ── 조향 계산 내부 헬퍼 ──────────────────────────────

    def _update_steer_inputs(self, x_coords, target_y, slope, curvature):
        self.last_center_error = float(np.mean(x_coords) - self.target)
        self.last_slope = float(slope)
        self.last_curvature = float(curvature)
        self.last_target_y_px = float(target_y)
        self.last_lookahead_px = float(self.bev_height - target_y)

    # 중심선 다항식 계산
    def _get_polyfit_m(self, target_y_ratio=0.55):
        x_coords, y_coords = [], []
        for i in self.target_indxs:
            x_coords.append((self.left_centers[i] + self.right_centers[i]) / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * target_y_ratio
        m = -(2 * a * target_y + b)
        return m, x_coords, y_coords

    # 직선 조향 계산
    def _straight_calc(self):
        m, x_coords, _ = self._get_polyfit_m(0.58)
        if np.abs(m) >= 0.4:
            self.steer = float(np.clip(m * 170, -99, 99)); self.max_step = 70; self.alpha = 0.75
        elif np.abs(m) >= 0.3:
            self.steer = float(np.clip(m * 160, -95, 95)); self.max_step = 65; self.alpha = 0.75
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -95, 95)); self.max_step = 60; self.alpha = 0.75
        elif np.abs(m) >= 0.1:
            self.steer = float(np.clip(m * 150, -90, 90)); self.max_step = 55; self.alpha = 0.75
        elif np.abs(m) >= 0.07:
            self.steer = float(np.clip(m * 100, -85, 85)); self.max_step = 40; self.alpha = 0.75
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            s = (m * 180.0) + (error * 0.08)
            self.steer = 0.0 if abs(error) < 30 else error / 7
            self.steer = float(np.clip(self.steer, -3, 3))
        else:
            self.steer = float(m * 20 / 0.1); self.max_step = 40

    # 직선 조향 계산
    def _curve_calc(self):
        m, x_coords, _ = self._get_polyfit_m(0.6)
        if np.abs(m) >= 0.4:
            self.steer = float(np.clip(m * 170, -99, 99)); self.max_step = 70; self.alpha = 0.75
        elif np.abs(m) >= 0.3:
            self.steer = float(np.clip(m * 160, -99, 99)); self.max_step = 60; self.alpha = 0.75
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -95, 95)); self.max_step = 60; self.alpha = 0.75
        elif np.abs(m) >= 0.1:
            self.steer = float(np.clip(m * 150, -90, 90)); self.max_step = 55; self.alpha = 0.75
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            self.steer = 0.0 if abs(error) < 40 else error / 7
            self.steer = float(np.clip(self.steer, -99, 99))
        else:
            self.steer = float(m * 15 / 0.1); self.max_step = 40

    # ── 모드별 조향 함수 ───────────────────────────────────────────────

    # 직선 주행
    def straight(self): #left-yellow / right-white
        x_coords, y_coords = [], []
        for i in self.target_indxs:
            x_coords.append(self.left_centers[i] + self.lane_width / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * 0.58
        m = -(2 * a * target_y + b)
        self._update_steer_inputs(x_coords, target_y, m, a)
        if np.abs(m) >= 0.4:
            self.steer = float(np.clip(m * 170, -99, 99)); self.max_step = 70; self.alpha = 0.75
        elif np.abs(m) >= 0.3:
            self.steer = float(np.clip(m * 160, -95, 95)); self.max_step = 65; self.alpha = 0.75
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -95, 95)); self.max_step = 60; self.alpha = 0.75
        elif np.abs(m) >= 0.1:
            self.steer = float(np.clip(m * 150, -90, 90)); self.max_step = 55; self.alpha = 0.75
        elif np.abs(m) >= 0.07:
            self.steer = float(np.clip(m * 100, -85, 85)); self.max_step = 40; self.alpha = 0.75
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            s = (m * 180.0) + (error * 0.08)
            self.steer = 0.0 if abs(error) < 30 else error / 7
            self.steer = float(np.clip(self.steer, -3, 3))
        else:
            self.steer = float(m * 20 / 0.1); self.max_step = 40

        self.lane_state = True
        self.publish_steer()
        self.publish_bool()

    # 반대 차선 직선 주행 
    def straight2(self): #left-white / right-yellow
        x_coords, y_coords = [], []
        for i in self.target_indxs:
            x_coords.append(self.right_centers[i] - self.lane_width / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * 0.58
        m = -(2 * a * target_y + b)
        self._update_steer_inputs(x_coords, target_y, m, a)
        if np.abs(m) >= 0.4:
            self.steer = float(np.clip(m * 170, -99, 99)); self.max_step = 70; self.alpha = 0.75
        elif np.abs(m) >= 0.3:
            self.steer = float(np.clip(m * 160, -95, 95)); self.max_step = 65; self.alpha = 0.75
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -95, 95)); self.max_step = 60; self.alpha = 0.75
        elif np.abs(m) >= 0.1:
            self.steer = float(np.clip(m * 150, -90, 90)); self.max_step = 55; self.alpha = 0.75
        elif np.abs(m) >= 0.07:
            self.steer = float(np.clip(m * 100, -85, 85)); self.max_step = 40; self.alpha = 0.75
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            s = (m * 180.0) + (error * 0.08)
            self.steer = 0.0 if abs(error) < 30 else error / 7
            self.steer = float(np.clip(self.steer, -3, 3))
        else:
            self.steer = float(m * 20 / 0.1); self.max_step = 40

        self.lane_state = True
        self.publish_steer()
        self.publish_bool()

    # 곡선 주행
    def curve(self):
        self._curve_calc()
        self.lane_state = True
        self.publish_steer()
        self.publish_bool()

    # 어린이 보호구역 주행 
    def school_zone(self):
        self.max_step = 25
        self.alpha    = 0.3
        x_coords, y_coords = [], []
        for i in self.target_indxs:
            x_coords.append((self.left_centers[i] + self.right_centers[i]) / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        x_coords, y_coords = self.filter_straight_outliers(x_coords, y_coords,
                                                           slope_band=0.15)
        if len(x_coords) < 3:
            self.lane_state = True
            self.publish_steer()
            self.publish_bool()
            return
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * 0.85
        m = -(2 * a * target_y + b)
        if -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            self.steer = 0.0 if abs(error) < 40 else error / 7
            self.steer = float(np.clip(self.steer, -10, 10))
        else:
            self.steer = float(np.clip(m * 10, -10, 10))
            self.max_step = 30
        self.lane_state = True
        self.publish_steer()
        self.publish_bool()

    # 왼쪽 차선만 검출된 경우
    def only_left(self):
        only_left_indxs = [2, 3, 5, 6, 7]
        x_coords, y_coords = [], []
        for i in only_left_indxs:
            x_coords.append(self.left_centers[i] + self.lane_width / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * 0.42
        m = -(2 * a * target_y + b)
        self._update_steer_inputs(x_coords, target_y, m, a)
        if m <= -0.35:
            self.steer = float(np.clip(m * 170, -99, 0));  self.max_step = 85; self.alpha = 0.85
        elif m >= 0.25:
            self.steer = float(np.clip(m * 165, 0,  99));  self.max_step = 80; self.alpha = 0.85
        elif m <= -0.25:
            self.steer = float(np.clip(m * 160, -99, 0));  self.max_step = 75; self.alpha = 0.85
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -99, 99)); self.max_step = 75; self.alpha = 0.85
        elif np.abs(m) >= 0.07:
            self.steer = float(np.clip(m * 150, -92, 92)); self.max_step = 68; self.alpha = 0.85
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            self.steer = 0.0 if abs(error) < 30 else error / 7
            self.steer = float(np.clip(self.steer, -99, 99))
        else:
            self.steer = float(m * 20 / 0.1); self.max_step = 40
        self.lane_state = True
        self.publish_steer()
        self.publish_bool()

    # 오른쪽 차선만 검출된 경우
    def only_right(self):
        only_right_indxs = [2, 3, 5, 6, 7]
        x_coords, y_coords = [], []
        for i in only_right_indxs:
            x_coords.append(self.right_centers[i] - self.lane_width / 2)
            y_coords.append(self.bev_height
                            - (i * self.window_height)
                            - (self.window_height / 2))
        a, b, c = np.polyfit(y_coords, x_coords, 2)
        target_y = self.bev_height * 0.42
        m = -(2 * a * target_y + b)
        self._update_steer_inputs(x_coords, target_y, m, a)
        if m >= 0.35:
            self.steer = float(np.clip(m * 170, 0,  99));  self.max_step = 85; self.alpha = 0.85
        elif m <= -0.25:
            self.steer = float(np.clip(m * 165, -99, 0));  self.max_step = 80; self.alpha = 0.85
        elif m >= 0.25:
            self.steer = float(np.clip(m * 160, 0,  95));  self.max_step = 70; self.alpha = 0.85
        elif np.abs(m) >= 0.2:
            self.steer = float(np.clip(m * 155, -99, 99)); self.max_step = 72; self.alpha = 0.85
        elif np.abs(m) >= 0.07:
            self.steer = float(np.clip(m * 150, -92, 92)); self.max_step = 64; self.alpha = 0.85
        elif -0.04 <= m <= 0.04:
            error = np.mean(x_coords) - self.target
            self.steer = 0.0 if abs(error) < 30 else error / 7
            self.steer = float(np.clip(self.steer, -99, 99))
        else:
            self.steer = float(m * 20 / 0.1); self.max_step = 40
        self.lane_state = True
        self.publish_steer()
        self.publish_bool()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()