# hand_trackpad_calibrated_double_pinch.py
# More consistent pinch + trackpad mouse:
# - Hold pinch to move cursor (relative, trackpad style)
# - Double pinch is easier: counts two "pinch pulses" within a time window = click
# - Calibration makes pinch consistent across distance, lighting, and hand size
#
# Install:
#   pip install opencv-python mediapipe pyautogui
#
# Controls:
#   o = calibrate OPEN hand (hold open hand for 2s)
#   p = calibrate PINCH (hold a comfortable pinch for 2s)
#   c = clear calibration
#   x = toggle invert X
#   q or Esc = quit
#
# Safety:
# - Move mouse to a screen corner to trigger PyAutoGUI failsafe

import cv2
import mediapipe as mp
import math
import time
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lm2(lm):
    return (lm.x, lm.y)


class CalibratedPinch:
    """
    Computes a scale-normalized pinch ratio from landmarks.
    Calibration sets ratio_scale so that your typical pinch ratio is around 0.8
    so a fixed threshold of 1.0 becomes easy and consistent.

    Pinch state:
    - pinch ON when ema_ratio < down_thresh (fixed at 1.0 after calibration)
    - pinch OFF when ema_ratio > up_thresh (derived from open-hand ratio)
    """

    def __init__(
        self,
        target_pinch_scaled=0.80,
        down_thresh=1.00,
        ema_alpha=0.30,
        stable_frames=3,
        closest_guard=0.85,
    ):
        self.target_pinch_scaled = target_pinch_scaled
        self.down_thresh = down_thresh
        self.ema_alpha = ema_alpha
        self.stable_frames = stable_frames
        self.closest_guard = closest_guard

        self.open_raw = []
        self.pinch_raw = []

        self.ratio_scale = 6.0  # placeholder until calibration
        self.up_thresh = 1.25   # placeholder until calibration
        self.calibrated = False

        self._ema = None
        self.is_pinching = False
        self._below = 0
        self._above = 0

        self.last_guard_ok = True
        self.last_ema = None

    def clear_calibration(self):
        self.open_raw = []
        self.pinch_raw = []
        self.calibrated = False
        self.ratio_scale = 6.0
        self.up_thresh = 1.25
        self._ema = None
        self.is_pinching = False
        self._below = 0
        self._above = 0

    def _hand_scale(self, lm):
        wrist = lm2(lm[0])
        middle_mcp = lm2(lm[9])
        index_mcp = lm2(lm[5])
        pinky_mcp = lm2(lm[17])
        index_tip = lm2(lm[8])

        palm_width = dist2(index_mcp, pinky_mcp)
        palm_height = dist2(wrist, middle_mcp)
        index_len = dist2(index_mcp, index_tip)
        return max(palm_width, palm_height, index_len, 1e-6)

    def _guard_ok(self, lm, pinch_d):
        thumb_tip = lm2(lm[4])
        middle_tip = lm2(lm[12])
        ring_tip = lm2(lm[16])
        pinky_tip = lm2(lm[20])

        nearest_other = min(
            dist2(thumb_tip, middle_tip),
            dist2(thumb_tip, ring_tip),
            dist2(thumb_tip, pinky_tip),
        )
        return pinch_d <= self.closest_guard * nearest_other

    def compute_raw_ratio(self, hand_landmarks):
        lm = hand_landmarks.landmark
        thumb_tip = lm2(lm[4])
        index_tip = lm2(lm[8])

        pinch_d = dist2(thumb_tip, index_tip)
        self.last_guard_ok = self._guard_ok(lm, pinch_d)

        scale = self._hand_scale(lm)
        raw_ratio = pinch_d / scale
        return raw_ratio

    def add_open_sample(self, raw_ratio):
        self.open_raw.append(raw_ratio)

    def add_pinch_sample(self, raw_ratio):
        self.pinch_raw.append(raw_ratio)

    def finalize_calibration(self):
        if len(self.open_raw) < 15 or len(self.pinch_raw) < 15:
            return False

        open_avg_raw = sum(self.open_raw) / len(self.open_raw)
        pinch_avg_raw = sum(self.pinch_raw) / len(self.pinch_raw)

        if pinch_avg_raw <= 1e-6:
            return False

        # Scale so pinch sits around target_pinch_scaled (usually 0.8),
        # making "ratio < 1.0" a comfortable pinch trigger.
        self.ratio_scale = self.target_pinch_scaled / pinch_avg_raw
        open_scaled = open_avg_raw * self.ratio_scale

        # Release threshold based on open hand, clamped to sane range.
        # Needs to be above 1.0 so it rearms reliably without flicker.
        self.up_thresh = max(1.15, min(2.20, 0.70 * open_scaled))

        self.calibrated = True

        # Reset state for clean start
        self._ema = None
        self.is_pinching = False
        self._below = 0
        self._above = 0
        return True

    def update(self, hand_landmarks):
        raw_ratio = self.compute_raw_ratio(hand_landmarks)

        scaled = raw_ratio * self.ratio_scale

        # If guard fails, treat as open to avoid random triggers
        scaled_for_state = scaled if self.last_guard_ok else 999.0

        if self._ema is None:
            self._ema = scaled_for_state
        else:
            a = self.ema_alpha
            self._ema = a * scaled_for_state + (1.0 - a) * self._ema

        self.last_ema = self._ema

        down = self.down_thresh
        up = self.up_thresh

        if not self.is_pinching:
            if self._ema < down:
                self._below += 1
                if self._below >= self.stable_frames:
                    self.is_pinching = True
                    self._above = 0
            else:
                self._below = 0
        else:
            if self._ema > up:
                self._above += 1
                if self._above >= self.stable_frames:
                    self.is_pinching = False
                    self._below = 0
            else:
                self._above = 0

        return self.is_pinching, self._ema, down, up


class TrackpadCursor:
    """
    Relative cursor movement controller with strong anti-jitter.
    """

    def __init__(
        self,
        screen_w,
        screen_h,
        sensitivity=0.55,
        delta_alpha=0.10,
        deadzone=0.007,
        max_px_per_frame=16,
        invert_x=True,
        invert_y=False,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity
        self.delta_alpha = delta_alpha
        self.deadzone = deadzone
        self.max_px_per_frame = max_px_per_frame
        self.invert_x = invert_x
        self.invert_y = invert_y

        self.prev = None
        self.dx_ema = 0.0
        self.dy_ema = 0.0

    def reset(self):
        self.prev = None
        self.dx_ema = 0.0
        self.dy_ema = 0.0

    @staticmethod
    def _clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def update(self, nx, ny):
        if self.prev is None:
            self.prev = (nx, ny)
            return

        px, py = self.prev
        dx = nx - px
        dy = ny - py
        self.prev = (nx, ny)

        if abs(dx) < self.deadzone:
            dx = 0.0
        if abs(dy) < self.deadzone:
            dy = 0.0

        a = self.delta_alpha
        self.dx_ema = a * dx + (1.0 - a) * self.dx_ema
        self.dy_ema = a * dy + (1.0 - a) * self.dy_ema

        move_x = self.dx_ema * (self.sensitivity * self.screen_w)
        move_y = self.dy_ema * (self.sensitivity * self.screen_h)

        if self.invert_x:
            move_x = -move_x
        if self.invert_y:
            move_y = -move_y

        move_x = self._clamp(move_x, -self.max_px_per_frame, self.max_px_per_frame)
        move_y = self._clamp(move_y, -self.max_px_per_frame, self.max_px_per_frame)

        if move_x != 0.0 or move_y != 0.0:
            pyautogui.moveRel(move_x, move_y, duration=0)


class DoublePinchPulseClicker:
    """
    Double pinch click using pulses on the smoothed ratio:
    - A pulse is detected when ema dips below click_down
    - Rearm when ema rises above click_up
    - Two pulses within window => click
    """

    def __init__(self, window=1.30, min_gap=0.10, cooldown=0.35):
        self.window = window
        self.min_gap = min_gap
        self.cooldown = cooldown

        self.click_down = 0.95
        self.click_up = 1.10

        self.armed = True
        self.pulse_count = 0
        self.first_pulse_t = 0.0
        self.last_pulse_t = 0.0
        self.cooldown_until = 0.0

    def set_thresholds_from_pinch(self, down_thresh, up_thresh):
        # Easier than movement pinch:
        # click_down slightly below pinch down threshold
        # click_up slightly above pinch up threshold for clean rearm
        self.click_down = max(0.60, down_thresh - 0.05)
        self.click_up = max(self.click_down + 0.10, min(2.50, up_thresh + 0.10))

    def reset(self):
        self.armed = True
        self.pulse_count = 0
        self.first_pulse_t = 0.0
        self.last_pulse_t = 0.0

    def update(self, ema_ratio, guard_ok):
        now = time.time()

        if ema_ratio is None or not guard_ok:
            return False
        if now < self.cooldown_until:
            return False

        if self.pulse_count > 0 and (now - self.first_pulse_t) > self.window:
            self.reset()

        if (not self.armed) and (ema_ratio > self.click_up):
            self.armed = True

        if self.armed and (ema_ratio < self.click_down):
            if (now - self.last_pulse_t) >= self.min_gap:
                self.pulse_count += 1
                self.last_pulse_t = now
                if self.pulse_count == 1:
                    self.first_pulse_t = now
                elif self.pulse_count >= 2:
                    pyautogui.click()
                    self.cooldown_until = now + self.cooldown
                    self.reset()
                    return True
            self.armed = False

        return False


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try index 1 if you have multiple cameras.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    pinch = CalibratedPinch(
        target_pinch_scaled=0.80,
        down_thresh=1.00,
        ema_alpha=0.30,
        stable_frames=3,
        closest_guard=0.85,
    )

    screen_w, screen_h = pyautogui.size()
    trackpad = TrackpadCursor(
        screen_w,
        screen_h,
        sensitivity=0.55,
        delta_alpha=0.10,
        deadzone=0.007,
        max_px_per_frame=16,
        invert_x=True,
        invert_y=False,
    )

    clicker = DoublePinchPulseClicker(window=1.30, min_gap=0.10, cooldown=0.35)

    # Calibration capture state
    capture_mode = None  # "open" or "pinch"
    capture_until = 0.0
    capture_duration = 2.0

    prev_pinching = False
    prev_time = time.time()
    click_flash_until = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pinching_state = False
            ema_ratio = None
            down = pinch.down_thresh
            up = pinch.up_thresh
            guard_ok = False

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm = hand_lms.landmark
                index_tip = lm[8]
                nx, ny = index_tip.x, index_tip.y

                pinching_state, ema_ratio, down, up = pinch.update(hand_lms)
                guard_ok = pinch.last_guard_ok

                # Collect calibration samples if capturing
                now = time.time()
                if capture_mode is not None and now <= capture_until:
                    raw_ratio = pinch.compute_raw_ratio(hand_lms)
                    if pinch.last_guard_ok:
                        if capture_mode == "open":
                            pinch.add_open_sample(raw_ratio)
                        elif capture_mode == "pinch":
                            pinch.add_pinch_sample(raw_ratio)
                elif capture_mode is not None and time.time() > capture_until:
                    capture_mode = None
                    if pinch.finalize_calibration():
                        clicker.set_thresholds_from_pinch(pinch.down_thresh, pinch.up_thresh)

                # Movement while pinching
                if pinching_state:
                    trackpad.update(nx, ny)
                else:
                    if prev_pinching:
                        trackpad.reset()
                prev_pinching = pinching_state

                # Double pinch click
                if clicker.update(ema_ratio, guard_ok):
                    click_flash_until = time.time() + 0.4

                # Draw tips
                index_px = (int(nx * w), int(ny * h))
                thumb_px = (int(lm[4].x * w), int(lm[4].y * h))
                cv2.circle(frame, index_px, 11, (0, 255, 0), -1)
                cv2.circle(frame, thumb_px, 9, (255, 255, 0), -1)

            # UI
            cv2.putText(
                frame,
                f"PINCH MOVE: {'ON' if pinching_state else 'OFF'}  ratio < {down:.2f}  release > {up:.2f}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255) if pinching_state else (255, 255, 255),
                2,
            )

            cal_text = "CAL: ON" if pinch.calibrated else "CAL: OFF (press o then p)"
            cv2.putText(
                frame,
                f"{cal_text}  invert_x: {trackpad.invert_x}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if ema_ratio is not None:
                cv2.putText(
                    frame,
                    f"ratio(EMA): {ema_ratio:.3f}  guard: {'OK' if guard_ok else 'FAIL'}",
                    (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"double pinch pulses: < {clicker.click_down:.2f} then > {clicker.click_up:.2f}",
                    (10, 116),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            if capture_mode is not None:
                remaining = max(0.0, capture_until - time.time())
                cv2.putText(
                    frame,
                    f"CAPTURING {capture_mode.upper()}  {remaining:.1f}s",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 255, 0),
                    2,
                )

            if time.time() < click_flash_until:
                cv2.putText(
                    frame,
                    "CLICK",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            now2 = time.time()
            fps = 1.0 / max(1e-6, (now2 - prev_time))
            prev_time = now2
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Hand Trackpad (calibrated pinch)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("x"):
                trackpad.invert_x = not trackpad.invert_x
                trackpad.reset()

            if key == ord("c"):
                pinch.clear_calibration()

            if key == ord("o"):
                capture_mode = "open"
                capture_until = time.time() + capture_duration

            if key == ord("p"):
                capture_mode = "pinch"
                capture_until = time.time() + capture_duration

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
