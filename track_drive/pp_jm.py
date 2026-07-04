import math

#=============================================
# 파라미터 (튜닝 구간)
#=============================================
WHEELBASE        = 2.5
LOOKAHEAD_DIST   = 0.5
REAR_AXLE_OFFSET = 2.8
MAX_STEER        = 200.0
JUMP_THRESHOLD   = 1.55    # 목표점 x가 이 값 이상 점프하면 무시

class PurePursuit:
    def __init__(self):
        self.prev_x = None

    def get_steer(self, points):
        target = None
        best_diff = float('inf')

        #후륜축 중심으로 튜닝
        for (mid_x, mid_y) in points:
            real_x = mid_x
            real_y = mid_y + REAR_AXLE_OFFSET

        #차량 뒤에 있는 점 무시
            if real_y <= 0.0:
                continue

            distance = math.sqrt(real_x**2 + real_y**2)

            #ld 보다 낮은 거리 무시
            if distance < LOOKAHEAD_DIST:
                continue

            #ld와 가장 가까운 점 선택
            diff = abs(distance - LOOKAHEAD_DIST)
            if diff < best_diff:
                best_diff = diff
                target = (real_x, real_y, mid_x)

        if target is None:
            return 0.0

        real_x, real_y, raw_x = target

        # 목표점이 직전 대비 너무 튀면 무시
        if self.prev_x is not None:
            if abs(raw_x - self.prev_x) > JUMP_THRESHOLD:
                return None

        self.prev_x = raw_x

        #pure pursuit 알고리즘 공식
        alpha     = math.atan2(real_x, real_y)
        delta_rad = math.atan2(2.0 * WHEELBASE * math.sin(alpha), LOOKAHEAD_DIST)
        delta_deg = math.degrees(delta_rad)

        return max(-MAX_STEER, min(MAX_STEER, delta_deg))