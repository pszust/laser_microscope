
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage.filters import gaussian_filter, maximum_filter

def count_pixel_sum(image: Image):
    # Calculate the sum of all pixels in the image and return it
    total_sum = 0
    for i in range(image.height):
        for j in range(image.width):
            pixel = image.getpixel((j,i))
            total_sum += np.sum(pixel)
    print(f"External functione executed - image has {total_sum}!")
    return total_sum


def dechannel(im, ch):
    im[:,:,0] = im[:,:,ch]
    im[:,:,1] = im[:,:,ch]
    im[:,:,2] = im[:,:,ch]
    return im


def standard_map(imgs_min, imgs_pls, sigma = 2, ch = -1):
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
    
    img_min_g = (cv2.cvtColor(img_min_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    img_pls_g = (cv2.cvtColor(img_pls_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    
    for i in range(0, 2):
        img_min_g = gaussian_filter(img_min_g, sigma=sigma)
        img_pls_g = gaussian_filter(img_pls_g, sigma=sigma)
    return img_pls_g-img_min_g


def map_to_image(mapka, rng=0.1):
    cmap = plt.get_cmap('PiYG')
    image_map = cmap((mapka/rng+1)/2)[:,:,:3]
    return (image_map*255).astype(np.uint8)


def make_check_mask(shot_point, ang, arr, shot_size=110, check_size=40, span = 25):
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


def new_validity_array(mapka,
                       direction,
                       aop = [(400, 224), (450, 450)],  # area of operation = [(x, y), (w, h)]
                       shot_size=110,
                       check_size=50,
                       count=30,
                       thres=0.03,
                       angle_count=15):
    minx = aop[0][0]-int(aop[1][0]/2)+int(shot_size)
    miny = aop[0][1]-int(aop[1][1]/2)+int(shot_size)
    maxx = aop[0][0]+int(aop[1][0]/2)-int(shot_size)
    maxy = aop[0][1]+int(aop[1][1]/2)-int(shot_size)
        
    val_array = []
    for i in range(0, count):
        shot_point = (minx + int(np.random.random() * (maxx - minx)),
                      miny + int(np.random.random() * (maxy - miny)))
        
        # get shotmask ratios
        shot_mask = np.zeros((mapka.shape[0], mapka.shape[1]))
        shot_mask = cv2.circle(shot_mask, shot_point, shot_size, 1, -1)
        shot_arr = mapka.copy() * shot_mask
        sh_neg, sh_pos, sh_tot, sh_dead = get_nppixel_count(shot_arr, thres = thres)
        if direction == 1: gain = sh_neg/sh_tot
        if direction == -1: gain = sh_pos/sh_tot
        gain *= (1-(sh_dead/sh_tot))  # leave the dead region alone!

        # get checkmask ratios for each angle
        angles = np.linspace(int(360 / angle_count), 360, angle_count)
        for ang in angles:
            chm = make_check_mask(shot_point, ang, mapka.copy(), check_size=check_size)
            ch_arr = mapka.copy() * chm
            ch_neg, ch_pos, ch_tot, ch_dead = get_nppixel_count(ch_arr, thres = thres)
            if direction == 1: ratio = ch_pos/ch_tot
            if direction == -1: ratio = ch_neg/ch_tot
            ratio *= (1-(ch_dead/ch_tot))  # leave the dead alone!
            val_array.append([shot_point[0], shot_point[1], ang, gain, ratio, gain*ratio**2])
        
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


def decide_smelting(mapka, direction):
    pos_coverage, dead = check_mapka(mapka)  # TODO: set thersholds and size as variables
    kolo_thershold = 0.75  # TODO: set as variable
    if (direction == 1 and pos_coverage >= kolo_thershold) or (direction == 0 and pos_coverage <= 1-kolo_thershold):
        return {}

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


def calculate_pixel_position(x_start, y_start, pixel_size, cur_row, cur_col):
    x_pos = x_start + cur_col * pixel_size
    y_pos = y_start + cur_row * pixel_size
    return (x_pos, y_pos)


def check_mapka(mapka, size = 400, thresh = 0.005):
    width = size
    height = size

    mask = np.zeros(mapka.shape)

    cx = int(mask.shape[0]/2)
    cy = int(mask.shape[1]/2)
    mask = cv2.rectangle(mask, (int(cy-width/2), int(cx-height/2)), (int(cy+width/2), int(cx+height/2)), 1, -1)
    
    masked = mapka*mask
    
    pos = masked[masked > thresh].shape[0]
    neg = masked[masked < -thresh].shape[0]
    dead = (width*height)-masked[(abs(masked) > thresh)].shape[0]
    
    return pos/(pos+neg+1), dead/(width*height)