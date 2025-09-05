import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from scipy.ndimage.filters import gaussian_filter, maximum_filter


def count_pixel_sum(image: Image):
    # Calculate the sum of all pixels in the image and return it
    total_sum = 0
    for i in range(image.height):
        for j in range(image.width):
            pixel = image.getpixel((j, i))
            total_sum += np.sum(pixel)
    print(f"External functione executed - image has {total_sum}!")
    return total_sum


def dechannel(im, ch):
    im[:, :, 0] = im[:, :, ch]
    im[:, :, 1] = im[:, :, ch]
    im[:, :, 2] = im[:, :, ch]
    return im


def standard_map(imgs_min, imgs_pls, sigma=2, ch=-1):
    if type(imgs_min) is not np.ndarray:
        imgs_min = np.array(imgs_min)
        imgs_pls = np.array(imgs_pls)

    # img_min = np.mean(imgs_min, axis=0).astype(np.uint8)
    # img_pls = np.mean(imgs_pls, axis=0).astype(np.uint8)

    if ch != -1:
        img_min_g = dechannel(imgs_min.copy(), ch)
        img_pls_g = dechannel(imgs_pls.copy(), ch)
    else:
        img_min_g = imgs_min.copy()
        img_pls_g = imgs_pls.copy()

    img_min_g = (cv2.cvtColor(img_min_g, cv2.COLOR_BGR2GRAY)).astype(float) / 255
    img_pls_g = (cv2.cvtColor(img_pls_g, cv2.COLOR_BGR2GRAY)).astype(float) / 255

    for i in range(0, 2):
        img_min_g = gaussian_filter(img_min_g, sigma=sigma)
        img_pls_g = gaussian_filter(img_pls_g, sigma=sigma)
    return img_pls_g - img_min_g


def map_to_image(mapka, rng=0.1):
    cmap = plt.get_cmap("PiYG")
    image_map = cmap((mapka / rng + 1) / 2)[:, :, :3]
    return (image_map * 255).astype(np.uint8)


def make_check_mask(shot_point, ang, arr, shot_size=110, check_size=40, span=25):
    chm = np.zeros((arr.shape[0], arr.shape[1]))
    #     chm = cv2.circle(chm, shot_point, shot_size, 1, 2)
    cang = ang + 180
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    chm = cv2.circle(chm, (px, py), check_size, 1, -1)

    cang = ang + span + 180
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    chm = cv2.circle(chm, (px, py), int(check_size * 0.75), 1, -1)

    cang = ang - span + 180
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    chm = cv2.circle(chm, (px, py), int(check_size * 0.75), 1, -1)

    return chm


def new_validity_array(
    mapka,
    direction,
    aop=[(400, 224), (450, 450)],  # area of operation = [(x, y), (w, h)]
    shot_size=110,
    check_size=50,
    count=30,
    thres=0.03,
    angle_count=15,
):
    minx = aop[0][0] - int(aop[1][0] / 2) + int(shot_size)
    miny = aop[0][1] - int(aop[1][1] / 2) + int(shot_size)
    maxx = aop[0][0] + int(aop[1][0] / 2) - int(shot_size)
    maxy = aop[0][1] + int(aop[1][1] / 2) - int(shot_size)

    val_array = []
    for i in range(0, count):
        shot_point = (
            minx + int(np.random.random() * (maxx - minx)),
            miny + int(np.random.random() * (maxy - miny)),
        )

        # get shotmask ratios
        shot_mask = np.zeros((mapka.shape[0], mapka.shape[1]))
        shot_mask = cv2.circle(shot_mask, shot_point, shot_size, 1, -1)
        shot_arr = mapka.copy() * shot_mask
        sh_neg, sh_pos, sh_tot, sh_dead = get_nppixel_count(shot_arr, thres=thres)
        if direction == 1:
            gain = sh_neg / sh_tot
        if direction == 0:
            gain = sh_pos / sh_tot
        gain *= 1 - (sh_dead / sh_tot)  # leave the dead region alone!

        # get checkmask ratios for each angle
        angles = np.linspace(int(360 / angle_count), 360, angle_count)
        for ang in angles:
            chm = make_check_mask(shot_point, ang, mapka.copy(), check_size=check_size)
            ch_arr = mapka.copy() * chm
            ch_neg, ch_pos, ch_tot, ch_dead = get_nppixel_count(ch_arr, thres=thres)
            if direction == 1:
                ratio = ch_pos / ch_tot
            if direction == 0:
                ratio = ch_neg / ch_tot
            ratio *= 1 - (ch_dead / ch_tot)  # leave the dead alone!
            val_array.append([shot_point[0], shot_point[1], ang, gain, ratio, gain * ratio**2])

    val_array = np.array(val_array)
    val_array = np.flip(val_array[np.argsort(val_array[:, 5])], axis=0)
    return val_array


def show_shot_asrgb(target, mapka_rgb):
    drawn_shot = mapka_rgb.copy()
    ang = target["angle"] + 180
    center = (int(target["posx"]), int(target["posy"]))
    dir_x = int(target["posx"] - target["size"] * np.cos(ang * np.pi / 180))
    dir_y = int(target["posy"] - target["size"] * np.sin(-ang * np.pi / 180))
    drawn_shot = cv2.arrowedLine(drawn_shot, center, (dir_x, dir_y), (255, 24, 24), 3)
    drawn_shot = cv2.circle(drawn_shot, center, radius=target["size"], color=(0, 0, 0), thickness=2)
    return drawn_shot


def get_nppixel_count(masked_map, thres=0.01):
    neg = np.where(masked_map < -thres, 1, 0)
    pos = np.where(masked_map > thres, 1, 0)
    tot = np.where(masked_map != 0, 1, 0)
    dead = np.where((masked_map > -thres) & (masked_map < thres) & (masked_map != 0), 1, 0)
    return neg.sum(), pos.sum(), tot.sum(), dead.sum()


def is_dict_empty(dictionary: dict):
    if dictionary:
        return 0
    return 1


def get_time_from_target(target):
    return target["duration"]


def _is_melt_condition(direction: int, pos_coverage: float, thr: float) -> bool:
    return (direction == 1 and pos_coverage >= thr) or (direction == 0 and pos_coverage <= 1 - thr)


def decide_smelting(mapka, direction, work_size=400, det_thr=0.005, thr1=0.5, thr2=0.9):
    pos_coverage, dead = check_mapka(mapka, size=work_size, thresh=det_thr)
    if _is_melt_condition(direction, pos_coverage, thr2):
        return {"anim_path": "IS_DONE"}
    elif _is_melt_condition(direction, pos_coverage, thr1):
        # kolowa animacja
        val_arr = new_validity_array(mapka, direction)
        kolo_size = 150
        kolo_dur = 12
        kolo_path = "shining-moon3.anim"
        target = {
            "posx": val_arr[0, 0],
            "posy": val_arr[0, 1],
            "angle": val_arr[0, 2],
            "size": kolo_size,
            "duration": kolo_dur,
            "anim_path": kolo_path,
        }
        return target
    else:
        # kwadratowa animacja
        kolo_size = 150
        kolo_dur = 12
        kolo_path = "shining-moon3.anim"
        target = {
            "posx": 100,
            "posy": 250,
            "angle": 0,
            "size": kolo_size,
            "duration": kolo_dur,
            "anim_path": kolo_path,
        }
        return target


def get_value_from_dict(dictionary, value):
    return dictionary.get(value, "NO_VALUE")


def calculate_pixel_position(x_start, y_start, pixel_size, cur_row, cur_col):
    x_pos = x_start + cur_col * pixel_size
    y_pos = y_start + cur_row * pixel_size
    return (x_pos, y_pos)


def check_mapka(mapka, size=400, thresh=0.005):
    width = size
    height = size

    mask = np.zeros(mapka.shape)

    cx = int(mask.shape[0] / 2)
    cy = int(mask.shape[1] / 2)
    mask = cv2.rectangle(
        mask,
        (int(cy - width / 2), int(cx - height / 2)),
        (int(cy + width / 2), int(cx + height / 2)),
        1,
        -1,
    )

    masked = mapka * mask

    pos = masked[masked > thresh].shape[0]
    neg = masked[masked < -thresh].shape[0]
    dead = (width * height) - masked[(abs(masked) > thresh)].shape[0]

    return pos / (pos + neg + 1), dead / (width * height)


def pack_variables(*args) -> tuple:
    result = tuple(arg for arg in args)
    return result


def consult_pattern(pattern_array: np.ndarray, x: int, y: int) -> str:
    y_shape, x_shape = pattern_array.shape
    if x >= x_shape:
        result = "NEXT_ROW"
    elif y >= y_shape:
        result = "DONE"
    else:
        result = "WORK"
    return result


def read_from_array(pattern_array: np.ndarray, x: int, y: int) -> int:
    return pattern_array[y][x]


def get_delta_move_on_array(
    size_mm: float, c_x: int, c_y: int, old_x: int, old_y: int
) -> tuple[float, float]:
    dx_mm = (old_x - c_x) * size_mm
    dy_mm = (old_y - c_y) * size_mm
    return dx_mm, dy_mm

    if color == 1:
        vars_xy = select_side(
            self.advanced_mapka448,
            1,
            check_size=int(self.extvars["pixel_check_size"]),
            side_thresh=float(self.extvars["side_threshold"]),
            thresh=float(self.extvars["check_threshold"]),
        )
    if color == 0:
        vars_xy = select_side(
            self.advanced_mapka448,
            -1,
            check_size=int(self.extvars["pixel_check_size"]),
            side_thresh=float(self.extvars["side_threshold"]),
            thresh=float(self.extvars["check_threshold"]),
        )


def make_var_list(cords):
    return [[60 * v / (490 / 2) for v in c] for c in cords]


def make_side_mask(c, s=100):
    width = s
    height = s

    maska = np.zeros((448, 800))
    cx = int(maska.shape[1] / 2) + 30 + c[0]
    cy = int(maska.shape[0] / 2) + c[1]
    maska = cv2.rectangle(
        maska,
        (int(cx - width / 2), int(cy - height / 2)),
        (int(cx + width / 2), int(cy + height / 2)),
        1,
        -1,
    )

    return maska


def make_coord_list(sqr_size=460):
    cords = []
    for i in range(0, 5):
        x = 0.5 * sqr_size
        y = (-0.5 * sqr_size) * i / 5
        cords.append([x, y])
        cords.append([-x, y])
        cords.append([-y, x])
        cords.append([y, -x])

        x = 0.5 * sqr_size
        y = (0.5 * sqr_size) * i / 5
        cords.append([x, y])
        cords.append([-x, y])
        cords.append([-y, x])
        cords.append([y, -x])
    return cords


def check_mapka_maske(mapka, mask, thresh=0.01):
    masked = mapka * mask

    pos = masked[masked > thresh].shape[0]
    neg = masked[masked < -thresh].shape[0]

    return pos / (pos + neg + 0.0001)


def select_side(mapka, target, check_size=100, side_thresh=0.8, thresh=0.01):
    """target = 1 or -1"""
    cords = make_coord_list()
    var_list = make_var_list(cords)
    posnegs = []
    for i in range(0, len(cords)):
        maska = make_side_mask(cords[i], s=check_size)
        posnegs.append(check_mapka_maske(mapka, maska, thresh=thresh))

    if target == 1:
        if max(posnegs) > side_thresh:
            ind = posnegs.index(max(posnegs))
            return var_list[ind]
        else:
            return [999, 999]
    if target == -1:
        if min(posnegs) < 1 - side_thresh:
            ind = posnegs.index(min(posnegs))
            return var_list[ind]
        else:
            return [999, 999]
