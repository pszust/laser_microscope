group_name_font = ("Segoe UI", 16)
subsystem_name_font = ("Segoe UI", 14, "bold")
info_label_font = ("Segoe UI", 12, "bold")
laser_on_color = "#772eff"
laser_off_color = "#5d615c"
info_label_color = "#5d615c"
main_loop_time = 175
con_colors = {"CONNECTED": "lime", "CONNECTING": "yellow", "NOT CONNECTED": "gray"}


class Device:
    USE_REAL_CAMERA = False
    USE_REAL_M30 = False
    USE_REAL_FLIPPERS = False
    USE_REAL_LABJACK = False
    USE_REAL_ROTATOR = False
    USE_REAL_RIGOL = False
    USE_REAL_HEATSTAGE = False


class ErrorMsg:
    err_var_missing = "Command: %s with args: %s - no %s in currently declared variables!"


class CamConsts:
    REAL_WIDTH = 1280
    REAL_HEIGHT = 960
    SHAPE = (REAL_WIDTH, REAL_HEIGHT, 3)
    ASPECT_RATIO = SHAPE[0] / SHAPE[1]
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = int(DISPLAY_WIDTH / ASPECT_RATIO)
    DISPLAY_SHAPE = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    BRUSH_SIZR_ARR = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        24,
        28,
        32,
        38,
        44,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
    ]


class ProjConsts:
    PROJ_IMG_SHAPE = (1024, 768, 3)
    SMALLER_SHAPE = (int(PROJ_IMG_SHAPE[0] / 4), int(PROJ_IMG_SHAPE[1] / 4), 3)


class LabJackConsts:
    SERIAL_NO = "55520124"
    MIN_POS = 0.0
    MAX_POS = 100.0
