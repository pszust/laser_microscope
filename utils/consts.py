group_name_font = ("Segoe UI", 16)
subsystem_name_font = ("Segoe UI", 14, "bold")
info_label_font = ("Segoe UI", 12, "bold")
laser_on_color = "#772eff"
laser_off_color = "#5d615c"
info_label_color = "#5d615c"
main_loop_time = 175
con_colors = {
    "CONNECTED": "lime",
    "CONNECTING": "yellow",
    "NOT CONNECTED": "gray"
    }


class ErrorMsg:
    err_var_missing = "Command: %s with args: %s - no %s in currently declared variables!"



class ProjConsts:
    PROJ_IMG_SHAPE = (1024, 768, 3)
    SMALLER_SHAPE = (int(PROJ_IMG_SHAPE[0]/4), int(PROJ_IMG_SHAPE[1]/4), 3)


class LabJackConsts:
    SERIAL_NO = "49499304"
    MIN_POS = 0.0
    MAX_POS = 100.0