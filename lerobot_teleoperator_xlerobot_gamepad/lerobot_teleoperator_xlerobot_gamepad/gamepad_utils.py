import logging
import warnings

import pygame

logger = logging.getLogger(__name__)

# Try to import SDL2's GameController API (preferred path).
# This routes raw evdev indices through SDL_GameControllerDB so the semantic
# button/axis names stay identical across Ubuntu 22 (hid-sony) and Ubuntu 24
# (hid-playstation) — i.e. across kernel HID drivers that otherwise present
# the PS5 controller with different raw index layouts.
try:
    from pygame._sdl2 import controller as _sdl_controller  # type: ignore
    _SDL_CONTROLLER_AVAILABLE = True
except ImportError:  # very old pygame; fall back to plain Joystick API
    _sdl_controller = None  # type: ignore
    _SDL_CONTROLLER_AVAILABLE = False


# :----- Naming conventions -----:
# - (l/r)s: left/right stick
# - (l/r)b: left/right bumper
# - (l/r)t: left/right trigger
# - dpad: d-pad
# - abxy: a/b/x/y buttons, for PS this is x/o/square/triangle
# - start: the tiny button on the top right
# - back: the tiny button on the top left
# - logo: logo button on the middle
#
# Note:
# 1. For sticks, 'ls' or 'rs' means left/right stick being pressed.
# 2. To specify the direction, add '_up/_down/_left/_right' to the key name, e.g., 'ls_up', 'dpad_up'.
# 3. To combine conditions, use '&' to connect them (no space around the '&').
#    E.g., 'ls&ls_up' means left stick being pressed and pushing up.
# 4. To negate a condition, add '!' before the condition, e.g., '!ls' means left stick not being pressed.


LEFT_KEYMAP: dict[str, str] = {
    # 左臂 XZ 控制 (左摇杆; 未按下)
    'left_arm.x+': '!ls&!rs&!lb&ls_up',
    'left_arm.x-': '!ls&!rs&!lb&ls_down',
    'left_arm.z+': '!ls&!rs&!lb&ls_right',
    'left_arm.z-': '!ls&!rs&!lb&ls_left',
    # 左臂 shoulder_pan (y) 和 pitch 控制 (LB 按下 + 左摇杆)
    'left_arm.pitch+': 'lb&ls_down',
    'left_arm.pitch-': 'lb&ls_up',
    'left_arm.y+': 'lb&ls_left',
    'left_arm.y-': 'lb&ls_right',
    # 左臂 wrist_yaw (new) 和 wrist_roll 控制 (LB 按下 + D-pad)
    'left_arm.roll+': 'lb&dpad_up',
    'left_arm.roll-': 'lb&dpad_down',
    'left_arm.yaw+': 'lb&dpad_left',
    'left_arm.yaw-': 'lb&dpad_right',
    # 左夹爪控制 (LT)
    'left_arm.gripper+': '!lb&lt',
    'left_arm.gripper-': 'lb&lt'
}
RIGHT_KEYMAP = {
    # 右臂 XZ 控制 (右摇杆; 未按下)
    'right_arm.x+': '!ls&!rs&!rb&rs_up',
    'right_arm.x-': '!ls&!rs&!rb&rs_down',
    'right_arm.z+': '!ls&!rs&!rb&rs_right',
    'right_arm.z-': '!ls&!rs&!rb&rs_left',
    # 右臂 shoulder_pan (y) 和 pitch 控制 (RB 按下 + 左摇杆)
    'right_arm.pitch+': 'rb&rs_down',
    'right_arm.pitch-': 'rb&rs_up',
    'right_arm.y+': 'rb&rs_left',
    'right_arm.y-': 'rb&rs_right',
    # 右臂 wrist_yaw (new) 和 wrist_roll 控制 (RB 按下 + abxy)
    'right_arm.roll+': 'rb&y',
    'right_arm.roll-': 'rb&a',
    'right_arm.yaw+': 'rb&x',
    'right_arm.yaw-': 'rb&b',
    # 右夹爪控制 (RT)
    'right_arm.gripper+': '!rb&rt',
    'right_arm.gripper-': 'rb&rt'
}
HEAD_KEYMAP = {
    # 头部电机控制 Xbox: (x, y, a, b); PS5: (s, t, x, o)
    "head.yaw+": '!rb&x',
    "head.yaw-": '!rb&b',
    "head.pitch+": '!rb&a',
    "head.pitch-": '!rb&y'
}
BASE_KEYMAP = {
    # 底盘控制
    'base.forward': '!lb&!rs&dpad_up',
    'base.backward': '!lb&!rs&dpad_down',
    'base.left': 'rs&!lb&dpad_left',
    'base.right': 'rs&!lb&dpad_right',
    'base.rotate_left': '!lb&!rs&dpad_left',
    'base.rotate_right': '!lb&!rs&dpad_right',
    'base.speed_up': 'back' # Xbox: back; PS5: create/options button; the tiny button on the left
}

RESET_KEYMAP = {
    'back_robot_to_zero': 'start', # Xbox: start; PS5: share button; the tiny button on the right
    'exit_teleop': 'logo'
}

RECORD_KEYMAP = {
    'exit_early': 'rs&ls_right',
    'rerecord_episode': 'rs&ls_left',
    'stop_recording': 'rs&ls_up',
}

ALL_KEYMAP = {
    **LEFT_KEYMAP,
    **RIGHT_KEYMAP,
    **HEAD_KEYMAP,
    **BASE_KEYMAP,
    **RESET_KEYMAP,
    **RECORD_KEYMAP,
}

def print_decode_keymap(keymap: dict[str, str]):
    words = {
        'ls': 'left stick press',
        'rs': 'right stick press',
        'lb': 'left bumper',
        'rb': 'right bumper',
        'lt': 'left trigger',
        'rt': 'right trigger',
        'start': 'start button',
        'back': 'back button',
        'logo': 'logo button',
        'dpad': 'd-pad',
        '!': 'not ',
        '_': ' '
    }
    print("\033[92m")
    print("*-----------------------------*")
    print("*      Control Key Map        *")
    print("*-----------------------------*")
    print("*   [Action] -> Key Mapping   *")
    print("*-----------------------------*")
    print("\033[0m")
    for action, control in keymap.items():
        conditions = control.split('&')
        for i, condition in enumerate(conditions):
            condition = condition.replace('ls_', 'left stick ')
            condition = condition.replace('rs_', 'right stick ')
            for c, n in words.items():
                condition = condition.replace(c, n)
            conditions[i] = condition
        _conn = ' \033[4mand\033[0m '
        print(f"\033[94m[{action:^10}]\033[0m {_conn.join(conditions):^15}")


def decode_key(gamepad, code: str) -> bool:
    conditions = code.split('&')
    state = True
    for condition in conditions:
        negate = condition.startswith('!')
        condition = condition.lstrip('!')
        if condition in ['ls', 'rs', 'lb', 'rb', 'lt', 'rt', 'start', 'back', 'logo', 'a', 'b', 'x', 'y']:
            state_i = gamepad.get_button(condition)
        elif condition.startswith('ls') or condition.startswith('rs'):
            stick = gamepad.get_left_stick() if condition.startswith('ls') else gamepad.get_right_stick()
            if condition.endswith('_up'):
                state_i= stick[1] < -0.5
            elif condition.endswith('_down'):
                state_i= stick[1] > 0.5
            elif condition.endswith('_left'):
                state_i= stick[0] < -0.5
            elif condition.endswith('_right'):
                state_i= stick[0] > 0.5
        elif condition.startswith('dpad'):
            dpad = gamepad.get_dpad()
            if condition.endswith('_up'):
                state_i= dpad[1] == 1
            elif condition.endswith('_down'):
                state_i= dpad[1] == -1
            elif condition.endswith('_left'):
                state_i= dpad[0] == -1
            elif condition.endswith('_right'):
                state_i= dpad[0] == 1
        else:
            raise ValueError(f"Invalid key: {condition}")
        if negate:
            state_i = not state_i
        state &= state_i
    return state

def get_gamepad_states(gamepad, keymap: dict[str, str]) -> dict[str, bool]:
    gamepad.update()
    states = dict.fromkeys(keymap.keys(), False)
    for action, control in keymap.items():
        states[action] = decode_key(gamepad, control)
    return states


# Semantic button names that SDLGamepad exposes through get_button().
_SEMANTIC_BUTTONS = (
    "a", "b", "x", "y",
    "lb", "rb", "lt", "rt",
    "start", "back", "logo",
    "ls", "rs",
)

# SDL GameController button-constant names for each semantic button.
# Resolved via getattr(pygame, NAME) at runtime to stay tolerant of pygame
# builds that don't expose every constant. Triggers are handled as axes.
_SDL_BUTTON_CONST_NAMES = {
    "a":     "CONTROLLER_BUTTON_A",
    "b":     "CONTROLLER_BUTTON_B",
    "x":     "CONTROLLER_BUTTON_X",
    "y":     "CONTROLLER_BUTTON_Y",
    "lb":    "CONTROLLER_BUTTON_LEFTSHOULDER",
    "rb":    "CONTROLLER_BUTTON_RIGHTSHOULDER",
    "back":  "CONTROLLER_BUTTON_BACK",
    "start": "CONTROLLER_BUTTON_START",
    "logo":  "CONTROLLER_BUTTON_GUIDE",
    "ls":    "CONTROLLER_BUTTON_LEFTSTICK",
    "rs":    "CONTROLLER_BUTTON_RIGHTSTICK",
}

# Joystick-API fallback layouts (used only when the SDL GameController API
# can't claim the device — rare for a real PS5 DualSense). Add new entries
# here if you encounter an unrecognized layout; pick which one to use by
# matching against joystick.get_name() in _detect_joystick_layout.
_JOYSTICK_LAYOUTS: dict[str, dict] = {
    # Legacy hid-sony / generic mapping (matches the original hand-coded layout).
    "sony_legacy": {
        "btn": {
            "a": 0, "b": 1, "y": 2, "x": 3,
            "lb": 4, "rb": 5,
            "start": 8, "back": 9, "logo": 10,
            "ls": 11, "rs": 12,
        },
        # Trigger axes: in the legacy driver these range from -1 (released)
        # to +1 (fully pressed); we keep the original ">0.5" threshold.
        "lt_axis": 2,
        "rt_axis": 5,
        "ls_axes": (0, 1),
        "rs_axes": (3, 4),
    },
}

# Threshold for treating analog triggers as a digital press.
_TRIGGER_THRESHOLD = 0.5
# SDL Controller axes are signed int16 — normalize to [-1, 1] / [0, 1].
_SDL_AXIS_DENOM = 32767.0


class SDLGamepad:
    """Generic SDL/pygame game-controller wrapper, robust across Linux distros / HID drivers.

    Works with any SDL-recognised controller (PS5 DualSense, Xbox, generic
    pads); the name reflects the SDL GameController backend, not a specific pad.

    Why this class exists:
        The raw button/axis indices exposed by ``pygame.joystick`` depend on
        which kernel HID driver claims the controller. Ubuntu 22.04 typically
        uses the older ``hid-sony`` mapping, while Ubuntu 24.04 uses the
        newer ``hid-playstation`` driver — same pygame, same SDL, but
        different raw indices. Hard-coded index tables therefore break when
        you move between the two.

        This class prefers SDL2's GameController API (via
        ``pygame._sdl2.controller``), which routes everything through
        ``SDL_GameControllerDB`` and gives stable semantic names. As a
        fallback it uses the Joystick API with a name-detected layout table.

    Public surface preserved from the original implementation:
        ``connect``, ``disconnect``, ``is_connected``, ``update``, ``reset``,
        ``get_button(name)``, ``get_left_stick()``, ``get_right_stick()``,
        ``get_dpad()``.
    """

    def __init__(self, id: int = 0):
        self.id = id
        self._mode: str | None = None  # 'controller' or 'joystick'
        self._controller = None
        self._joystick = None
        self._joystick_layout: str | None = None
        self.reset()

    def is_connected(self) -> bool:
        if self._mode == "controller" and self._controller is not None:
            try:
                return bool(self._controller.attached())
            except Exception:
                return False
        if self._mode == "joystick" and self._joystick is not None:
            return bool(self._joystick.get_init())
        return False

    def connect(self) -> None:
        pygame.init()
        pygame.joystick.init()

        # ---- Primary: SDL GameController API (driver-agnostic mapping) ----
        if _SDL_CONTROLLER_AVAILABLE:
            try:
                _sdl_controller.init()
                if (
                    _sdl_controller.get_count() > self.id
                    and _sdl_controller.is_controller(self.id)
                ):
                    self._controller = _sdl_controller.Controller(self.id)
                    self._mode = "controller"
                    name = getattr(self._controller, "name", "<unknown>")
                    logger.info(
                        "SDLGamepad: using SDL GameController API "
                        f"(id={self.id}, name={name!r})"
                    )
                    return
                logger.warning(
                    f"SDLGamepad: device {self.id} is not registered as an SDL "
                    "GameController; falling back to Joystick API."
                )
            except Exception as e:
                logger.warning(
                    f"SDLGamepad: SDL GameController init failed ({e!r}); "
                    "falling back to Joystick API."
                )
        else:
            logger.warning(
                "SDLGamepad: pygame._sdl2.controller unavailable; "
                "falling back to Joystick API."
            )

        # ---- Fallback: raw Joystick API with layout auto-detect ----
        if pygame.joystick.get_count() <= self.id:
            logger.error("No gamepad detected. Please connect a gamepad and try again.")
            return
        self._joystick = pygame.joystick.Joystick(self.id)
        self._joystick.init()
        self._mode = "joystick"
        self._joystick_layout = self._detect_joystick_layout(self._joystick)
        guid = getattr(self._joystick, "get_guid", lambda: "?")()
        logger.info(
            "SDLGamepad: using Joystick API fallback "
            f"(id={self.id}, name={self._joystick.get_name()!r}, "
            f"guid={guid}, layout={self._joystick_layout!r})"
        )

    @staticmethod
    def _detect_joystick_layout(joystick) -> str:
        """Pick a fallback layout from joystick name. Extend as needed."""
        name = (joystick.get_name() or "").lower()
        # Only the legacy layout is shipped today; the SDL Controller API
        # handles essentially every real-world PS5 case, so the fallback
        # only fires on unusual setups. If you ever need a 'dualsense' or
        # bluetooth-specific layout, add it to _JOYSTICK_LAYOUTS and branch
        # here on the name.
        _ = name  # reserved for future name-based detection
        return "sony_legacy"

    def disconnect(self) -> None:
        if self._mode == "controller" and self._controller is not None:
            try:
                self._controller.quit()
            except Exception:
                pass
            self._controller = None
        if self._mode == "joystick" and self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
            self._joystick = None
        if _SDL_CONTROLLER_AVAILABLE:
            try:
                _sdl_controller.quit()
            except Exception:
                pass
        pygame.quit()
        self._mode = None

    def reset(self):
        self._buttons: dict[str, bool] = {k: False for k in _SEMANTIC_BUTTONS}
        self._left_stick: tuple[float, float] = (0.0, 0.0)
        self._right_stick: tuple[float, float] = (0.0, 0.0)
        self._dpad: tuple[int, int] = (0, 0)
        self._lt_value: float = 0.0
        self._rt_value: float = 0.0

    def update(self):
        if self._mode is None:
            return  # not connected; pump would fail before pygame.init()
        pygame.event.pump()
        if self._mode == "controller":
            self._update_from_controller()
        elif self._mode == "joystick":
            self._update_from_joystick()

    # ---------- internal: SDL GameController path ----------

    def _update_from_controller(self):
        c = self._controller

        for name, const_name in _SDL_BUTTON_CONST_NAMES.items():
            const = getattr(pygame, const_name, None)
            if const is None:
                self._buttons[name] = False
                continue
            try:
                self._buttons[name] = bool(c.get_button(const))
            except Exception:
                self._buttons[name] = False

        lx = self._sdl_axis("CONTROLLER_AXIS_LEFTX")
        ly = self._sdl_axis("CONTROLLER_AXIS_LEFTY")
        rx = self._sdl_axis("CONTROLLER_AXIS_RIGHTX")
        ry = self._sdl_axis("CONTROLLER_AXIS_RIGHTY")
        self._left_stick = (lx / _SDL_AXIS_DENOM, ly / _SDL_AXIS_DENOM)
        self._right_stick = (rx / _SDL_AXIS_DENOM, ry / _SDL_AXIS_DENOM)

        # Triggers: SDL reports [0, 32767]; normalize to [0, 1].
        self._lt_value = max(0.0, self._sdl_axis("CONTROLLER_AXIS_TRIGGERLEFT") / _SDL_AXIS_DENOM)
        self._rt_value = max(0.0, self._sdl_axis("CONTROLLER_AXIS_TRIGGERRIGHT") / _SDL_AXIS_DENOM)
        self._buttons["lt"] = self._lt_value > _TRIGGER_THRESHOLD
        self._buttons["rt"] = self._rt_value > _TRIGGER_THRESHOLD

        right = self._sdl_btn("CONTROLLER_BUTTON_DPAD_RIGHT")
        left = self._sdl_btn("CONTROLLER_BUTTON_DPAD_LEFT")
        up = self._sdl_btn("CONTROLLER_BUTTON_DPAD_UP")
        down = self._sdl_btn("CONTROLLER_BUTTON_DPAD_DOWN")
        self._dpad = (int(right) - int(left), int(up) - int(down))

    def _sdl_axis(self, const_name: str) -> float:
        const = getattr(pygame, const_name, None)
        if const is None:
            return 0.0
        try:
            return float(self._controller.get_axis(const))
        except Exception:
            return 0.0

    def _sdl_btn(self, const_name: str) -> bool:
        const = getattr(pygame, const_name, None)
        if const is None:
            return False
        try:
            return bool(self._controller.get_button(const))
        except Exception:
            return False

    # ---------- internal: Joystick fallback path ----------

    def _update_from_joystick(self):
        j = self._joystick
        layout = _JOYSTICK_LAYOUTS[self._joystick_layout]
        nb = j.get_numbuttons()
        na = j.get_numaxes()

        def btn(idx):
            return bool(j.get_button(idx)) if idx is not None and idx < nb else False

        def axis(idx):
            return float(j.get_axis(idx)) if idx is not None and idx < na else 0.0

        btn_map = layout["btn"]
        for name in ("a", "b", "x", "y", "lb", "rb",
                     "start", "back", "logo", "ls", "rs"):
            self._buttons[name] = btn(btn_map.get(name))

        # Preserve original trigger semantics: raw axis > 0.5 means pressed.
        self._lt_value = axis(layout["lt_axis"])
        self._rt_value = axis(layout["rt_axis"])
        self._buttons["lt"] = self._lt_value > _TRIGGER_THRESHOLD
        self._buttons["rt"] = self._rt_value > _TRIGGER_THRESHOLD

        lx, ly = layout["ls_axes"]
        rx, ry = layout["rs_axes"]
        self._left_stick = (axis(lx), axis(ly))
        self._right_stick = (axis(rx), axis(ry))

        if j.get_numhats() > 0:
            hx, hy = j.get_hat(0)
            self._dpad = (int(hx), int(hy))
        else:
            self._dpad = (0, 0)

        # Sanity warning if the device shape is wildly different from what the
        # selected layout expects — surfaces "wrong layout" issues fast.
        if nb < max(btn_map.values()) + 1 or na < max(layout["lt_axis"], layout["rt_axis"]) + 1:
            warnings.warn(
                f"Joystick reports {nb} buttons / {na} axes — does not fit "
                f"layout {self._joystick_layout!r}. Add a matching entry to "
                "_JOYSTICK_LAYOUTS or use the SDL GameController path."
            )

    # ---------- public read API (unchanged signature) ----------

    def get_button(self, name: str) -> bool:
        if name not in self._buttons:
            raise ValueError(f"Invalid button: {name}")
        return self._buttons[name]

    def get_left_stick(self) -> tuple[float, float]:
        return self._left_stick

    def get_right_stick(self) -> tuple[float, float]:
        return self._right_stick

    def get_dpad(self) -> tuple[int, int]:
        return self._dpad


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG)

    print_decode_keymap(ALL_KEYMAP)
    gamepad = SDLGamepad()
    gamepad.connect()
    while gamepad.is_connected():
        states = get_gamepad_states(gamepad, ALL_KEYMAP)
        _states = {k: v for k, v in states.items() if v}
        print("Actions:", _states)
        print('\033[1A\033[K', end='')  # Move cursor up one line and clear the line
        time.sleep(0.1)
