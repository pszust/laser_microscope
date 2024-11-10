import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageChops
import os
import pandas as pd
# import sys, ftd2xx as ftd
import numpy as np
import serial
import sys
import glob
import time
from threading import Thread
import datetime
from scipy.ndimage.filters import gaussian_filter, maximum_filter
import pyvisa
import random
import imutils
from multiprocessing import Process
from multiprocessing import Value
from multiprocessing import RawArray
from multiprocessing import Event
from pyueye import ueye
import threading
from inputs import get_gamepad
import math

width, height = 800, 600
bar_height = 100
# cap = cv2.VideoCapture(1)
padd = 20
frame_x = 0
frame_y = 0
ell_offset_bottom = 0
proj_calib_frames = 25
# self.ell_offset_bottom = 146.25 # 62.5  # this is absolute angle at which relative angle is 90
data_dir = 'c:/Users/Laser/Documents/mikroskop_control/'
rmd = 'c:/Users/Laser/Documents/raw_mikro_data/'
results_dir = 'results/'

laser_control = False

# constants
inc_x = int(1024/5)
inc_y = int(768/5)  
rgb_weights = [0.2989, 0.5870, 0.1140]
calib_dots_dim = 4
ckernel = np.asarray([
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
]).astype(np.uint8)

img_shape = (448, 800, 3)

# constants needed for shot_array calculations
shot_cutout_x = 320
shot_cutout_y = 120

v_shot_x = 0
v_shot_y = 1
v_angle = 2
v_gain = 3  # count as in count of negative/positive values on the selected area of the map
v_ratio = 4  # check count, same as shot_count but for check map
v_validity = 5

scale_um = {'X5': 500, 'X10': 200, 'X20': 100, 'X50': 50, 'X100': 20}
scale_px = {'X5': 200, 'X10': 160, 'X20': 160, 'X50': 200, 'X100': 160}

eatwait_times_list = [10, 8, 5, 4, 3, 2, 1, 0.5, 0.25, 0]
eatspeed_value_list = [2, 5, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]

RAW = 0
MAP = 1
DYN = 2

laser_on_color = '#772eff'
laser_off_color = '#5d615c'

group_name_font = ('Segoe UI', 16)
subsystem_name_font = ('Segoe UI', 14, 'bold')

# counters 
projector_calib_c = calib_dots_dim**2+1
    
    
def round_to(val, to):
    val += to/2
    return int(val - val%to)
    

def sign(value):
    return abs(value)/value


def combine_by_date(date_str, obj = 'X5', sname = 'jamnik', sigma = 2, ch = -1, ranger = 0.1):
    scale_um = {'X5': 500, 'X10': 200, 'X20': 100, 'X50': 50, 'X100': 20}
    scale_px = {'X5': 200, 'X10': 160, 'X20': 160, 'X50': 200, 'X100': 160}
    # data_dir = 'raw_mikro_data/'
    
    files = [f for f in os.listdir(rmd) if f.find(date_str)>=0]
    
    frame_files = [f for f in files if f.find('frame_mapping')>=0]
    polars = list(set([f[f.rfind(' P')+1:f.rfind(' X')] for f in frame_files]))
    
    diffs = []
    for p in polars:
        val = float(p[1:])
        diffs.append(abs(90-val))
    diffs = list(set(diffs))
    
    if 0 in diffs: diffs.remove(0)
    if 90 in diffs: diffs.remove(90)
            
    for diff in diffs:
        pol = polars[0]
        cornames = [(rmd + f).replace(pol, '%s') for f in frame_files if f.find(pol)>=0]

        pol_min_name = 'P%2.1f'%(90-diff)
        pol_max_name = 'P%2.1f'%(90+diff)
        pol_diff_name = 'D%2.1f'%diff

        for cor in cornames:
            polmin = np.load(cor%pol_min_name)
            polpls = np.load(cor%pol_max_name)
            mapke = baseline_mapping(polpls, polmin, polmin*0, polmin*0, sigma = sigma, ch = ch)
            mapke = (255 * cmap_array(mapke, plt.get_cmap('PiYG'), rng=ranger)).astype(np.uint8)
            mapke = cv2.cvtColor(mapke, cv2.COLOR_BGR2RGB)
            np.save(cor%('%s'%pol_diff_name), mapke)
        polars.append('%s'%pol_diff_name)

        files = [f for f in os.listdir(rmd) if f.find(date_str)>=0]
        frame_files = [f for f in files if f.find('frame_mapping')>=0]
    
    for pol in polars:
        combine = combine_filelist([rmd + f for f in frame_files if f.find(pol)>=0], obj)

        scale_h = 40
        scale_w = scale_px[obj]

        scale_x = 20
        scale_y = combine.shape[0] - 20 - scale_h

        combine_scale = cv2.rectangle(combine.copy(), (scale_x-2, scale_y-2),
                                     (scale_x + scale_w+2, scale_y + scale_h+2),
                                     (209, 209, 209), -1)
        combine_scale = cv2.rectangle(combine_scale, (scale_x, scale_y),
                                     (scale_x + scale_w, scale_y + scale_h),
                                     (255, 255, 255), -1)

        font = cv2.FONT_HERSHEY_DUPLEX
        text = '%d um'%scale_um[obj]
        (label_width, label_height), baseline = cv2.getTextSize(text, font, 1, 1)
        org = (int(scale_x + 0.5*scale_w-label_width/2), int(scale_y + 0.5*scale_h+label_height/2))
        combine_scale = cv2.putText(combine_scale, text, org, font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(results_dir + 'maps/map_of_%s_%s_%s.jpg'%(sname, date_str, pol), combine_scale)

    # move files to new directory
    os.mkdir(rmd + 'raw_map_data/map_of_%s_%s_%s'%(sname, date_str, pol))
    for file in files:
        src = rmd + file
        dst = rmd + 'raw_map_data/map_of_%s_%s_%s/%s'%(sname, date_str, pol, file)
        os.rename(src, dst)

    
def get_coordinates_from_name(name):
    if name.find('x=') >= 0:
        x = name[name.find('x=')+2:name.find('y=')]
        y = name[name.find('y=')+2:name.find(').npy')]
        
    if name.find(' X') >= 0 and name.find(' Y') >= 0:
        x = name[name.find(' X')+2:name.find(' X')+2+3]
        y = name[name.find(' Y')+2:name.find(' Y')+2+3]
        
    return float(x), float(y)


def merge_arrays(arr1, crd1, arr2, crd2, crop=5):
    hor1 = crd1[1] + arr1.shape[1]
    ver1 = crd1[0] + arr1.shape[0]

    hor2 = crd2[1] + arr2.shape[1]
    ver2 = crd2[0] + arr2.shape[0]

    hor = int(max([hor1, hor2]))
    ver = int(max([ver1, ver2]))

    combine = np.zeros((ver, hor, 3), np.uint8)

    combine[crd1[0] + crop:arr1.shape[0] + crd1[0] - crop, crd1[1] +
            crop:arr1.shape[1] + crd1[1] - crop] = arr1[crop:-crop, crop:-crop]
    combine[crd2[0] + crop:arr2.shape[0] + crd2[0] - crop, crd2[1] +
            crop:arr2.shape[1] + crd2[1] - crop] = arr2[crop:-crop, crop:-crop]

    return combine


def combine_filelist(filelist, obj):
    for file in filelist:
        # get coordinates and recalculate to pixels
        x, y = get_coordinates_from_name(file)
        x = int(x*1000 * scale_px[obj]/scale_um[obj])
        y = int(y*1000 * scale_px[obj]/scale_um[obj])
        img = np.load(file)
#         img = cv2.putText(img, '%2.2f, %2.2f'%(x, y), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (235,20,20), 3, cv2.LINE_AA)

        if file == filelist[0]:
            combine = img.copy()
        else:
            combine = merge_arrays(combine, [0, 0], img, [y, x], crop = 5)
    return combine
    
    
def draw_rect(img, position, shape):
    shape = (int(shape[0] / 2), int(shape[1] / 2))
    img = cv2.rectangle(img, (position[0] - shape[0], position[1] - shape[1]),
                        (position[0] + shape[0], position[1] + shape[1]), (255, 255, 255),
                        -1)
    return img


def set_position(n, xgo, ygo):
    val = n
    if val < 0.25:
        xpos = (val - 0) / 0.25 * xgo
        ypos = 0
    if val >= 0.25 and val < 0.5:
        xpos = xgo
        ypos = (val - 0.25) / 0.25 * ygo
    if val >= 0.5 and val < 0.75:
        xpos = xgo - (val - 0.5) / 0.25 * xgo
        ypos = ygo
    if val >= 0.75:
        xpos = 0
        ypos = ygo - (val - 0.75) / 0.25 * ygo

    return (int(xpos), int(ypos))


def animate_square(n, shape, start_point, xrange, yrange):
    img = np.zeros((448, 800, 3), np.uint8)
    
    position = set_position(n, xrange, yrange)
    position = (position[0]+start_point[0], position[1]+start_point[1])
    img = draw_rect(img, position, shape)
    return img


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


def draw_check_mask(rgb, shot_point, ang, shot_size=110, check_size=50):
    cang = ang
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    rgb = cv2.circle(rgb, (px, py), check_size, (0, 0, 0), 2)

    cang = ang + 15
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    rgb = cv2.circle(rgb, (px, py), int(check_size * 0.7), (0, 0, 0), 2)

    cang = ang - 15
    px = int(shot_point[0] + shot_size * np.cos(cang * np.pi / 180))
    py = int(shot_point[1] + shot_size * np.sin(-cang * np.pi / 180))
    rgb = cv2.circle(rgb, (px, py), int(check_size * 0.7), (0, 0, 0), 2)

    return rgb


def show_shot_asrgb(val, mapka, shot_size=110, check_size=50, ranger = 0.1):
    rgb_mapka = (255 * cmap_array(mapka.copy(), diff_cmap, rng=ranger)).astype(
        np.uint8)
    rgb_mapka = cv2.circle(rgb_mapka, (int(val[v_shot_x]), int(val[v_shot_y])),
                           shot_size, (0, 0, 0), 2)

    ang = val[v_angle] + 180
    dir_x = int(val[v_shot_x] - shot_size * np.cos(ang * np.pi / 180))
    dir_y = int(val[v_shot_y] - shot_size * np.sin(-ang * np.pi / 180))
    rgb_mapka = cv2.arrowedLine(rgb_mapka,
                                (int(val[v_shot_x]), int(val[v_shot_y])),
                                (dir_x, dir_y), (255, 24, 24), 3)

    rgb_mapka = draw_check_mask(rgb_mapka,
                                (int(val[v_shot_x]), int(val[v_shot_y])),
                                ang,
                                shot_size=shot_size,
                                check_size=check_size)
    return rgb_mapka


def new_validity_array(mapka,
                       direction,
                       aop = [(400, 224), (450, 450)],  # area of operation = [(x, y), (w, h)]
                       shot_size=110,
                       check_size=50,
                       count=100,
                       thres=0.03,
                       angle_count=15):
    val_arr = []

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


def estimate_total(mapka):
    maska = np.zeros(mapka.shape)
    ox = int(shot_cutout_x / 2)
    oy = int(shot_cutout_y / 2)
    maska = cv2.rectangle(maska, (ox, oy), (maska.shape[1]-ox, maska.shape[0]-oy), 1, -1)
    masked_map = mapka.copy() * maska
    return get_nppixel_count(masked_map, thres = 0.01)
    

def get_nppixel_count(masked_map, thres=0.01):
    neg = np.where(masked_map < -thres, 1, 0)
    pos = np.where(masked_map > thres, 1, 0)
    tot = np.where(masked_map != 0, 1, 0)
    dead = np.where((masked_map > -thres) & (masked_map < thres) & (masked_map != 0), 1, 0)

    return neg.sum(), pos.sum(), tot.sum(), dead.sum()
    
    
def blur_crop(arr, sigma = 7, crop = 40):
    arr = gaussian_filter(arr, sigma=sigma)
    test = cv2.resize(arr, (arr.shape[1]+crop, arr.shape[0]+crop), interpolation = cv2.INTER_AREA)
    new = np.zeros(arr.shape, float)
    half_crop = int(crop/2)
    new = test[half_crop:-half_crop, half_crop:-half_crop]
    return new


def minimize_bisection(function, arguments = [], rng = [0, 1], jumps = 10):
    jump = abs(rng[1]-rng[0])/2  # current jump
    cp = (rng[0]+rng[1])/2  # current position
#     cr = function(cp)  # current result of the function
    for i in range(0, jumps):
        # calculate results on the left and right
        args = [cp-jump] + arguments
        rl = function(*args)
        args = [cp+jump] + arguments
        rr = function(*args)
#         print('jump = %d, cp = %f, rl=%f,rr=%f'%(i, cp, rl, rr))
        if rr < rl:
            cp += jump
        else:
            cp -= jump
        jump /= 2
    return cp


def estimate_gradient(img, sig = 300):
    blur = blur_crop(img, sigma = sig)
    return blur.max()/blur.min()


def check_factor(factor, image, grad):
    img_c = image.copy() - grad*factor + grad.mean()*factor
    return estimate_gradient(img_c)


def degradient(img, resize = 10, sig = 300, jumps = 8):
    sy = int(img.shape[0]/resize)
    sx = int(img.shape[1]/resize)
    resized = cv2.resize(img, (sx, sy), interpolation = cv2.INTER_AREA)
    grad = blur_crop(resized, sigma=int(sig/resize), crop = int(40/resize))
    
    factor = minimize_bisection(check_factor, arguments = [resized.copy(), grad], rng = [0, 4], jumps = jumps)
    # recalculate gradient using full unresized image
    grad = blur_crop(img, sigma=sig)
    return img.copy() - grad*factor + grad.mean()*factor
    

def advanced_map(img_minus, img_plus, sg = 4, degrad_size = 10, maxi = 2):
    # convert to grayscale
    arrPg = cv2.cvtColor(img_plus, cv2.COLOR_BGR2GRAY).astype(float)/255
    arrMg = cv2.cvtColor(img_minus, cv2.COLOR_BGR2GRAY).astype(float)/255
    
    # gauss if necessary
    if (sg>0):
        arrPg = gaussian_filter(arrPg, sigma=sg)
        arrMg = gaussian_filter(arrMg, sigma=sg)
    
    # degradient if necessary
    if (degrad_size>0):
        arrPg = degradient(arrPg, resize = degrad_size)
        arrMg = degradient(arrMg, resize = degrad_size)
        
    # maxi filter if necessary (good for eliminating birefringence)
    if maxi>0:
        arrPg = -maximum_filter(-arrPg, maxi)
        arrMg = -maximum_filter(-arrMg, maxi)
        
    # mapka
    mapka = arrPg-arrMg
    mapka = mapka-mapka.mean()
    
    # get domain threshold
    min1, max1 = get_minmax(get_distribution(mapka))
    ranger = max([abs(min1), max1])*0.3
    # np.save('temp.npy', mapka)
    
    return mapka, ranger


def baseline_mapping(img_min, img_pls, baseline_min, baseline_pls, sigma = 2, ch = 2):
    if ch != -1:
        img_min_g = dechannel(img_min.copy(), ch)
        img_pls_g = dechannel(img_pls.copy(), ch)

        b_min_g = dechannel(baseline_min.copy(), ch)
        b_pls_g = dechannel(baseline_pls.copy(), ch)
    else:
        img_min_g = img_min.copy()
        img_pls_g = img_pls.copy()
        
        b_min_g = baseline_min.copy()
        b_pls_g = baseline_pls.copy()
    
    img_min_g = (cv2.cvtColor(img_min_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    img_pls_g = (cv2.cvtColor(img_pls_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    
    b_min_g = (cv2.cvtColor(baseline_min, cv2.COLOR_BGR2GRAY)).astype(float)/255
    b_pls_g = (cv2.cvtColor(baseline_pls, cv2.COLOR_BGR2GRAY)).astype(float)/255
    
    for i in range(0, 2):
        img_min_g = gaussian_filter(img_min_g, sigma=sigma)
        img_pls_g = gaussian_filter(img_pls_g, sigma=sigma)
        b_min_g = gaussian_filter(b_min_g, sigma=sigma)
        b_pls_g = gaussian_filter(b_pls_g, sigma=sigma)
    
    img_min_g = img_min_g - b_min_g
    img_pls_g = img_pls_g - b_pls_g
    
    return img_pls_g-img_min_g
  

def dechannel(im, ch):
    im[:,:,0] = im[:,:,ch]
    im[:,:,1] = im[:,:,ch]
    im[:,:,2] = im[:,:,ch]
    return im


def get_linear_grad(arr, ch, norm = False, sav = 0):
    grad = ((arr).astype(float)/255)[:,:,ch].sum(axis = 0)/arr.shape[1]
    grad = grad[2:-2]
    if sav > 0:
        grad = sc.signal.savgol_filter(x=grad, window_length=sav, polyorder=3)
    
    if norm == True:
        grad -= grad.min()
        grad /= grad.max()
#         grad /= grad.mean()
    return grad


def mov_avg(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_bgr_channels(img_mins, img_plus, cutoff = 6, sg = 2, mx = 2):
    cm = dechannel(img_mins.copy(), 0)
    cp = dechannel(img_plus.copy(), 0)
    mapkaB, rangerB = advanced_map(cm, cp, sg = sg, maxi = mx, degrad_size = 0)
    lingradB = mapkaB.sum(axis = 0)/mapkaB.shape[0]

    cm = dechannel(img_mins.copy(), 1)
    cp = dechannel(img_plus.copy(), 1)
    mapkaG, rangerG = advanced_map(cm, cp, sg = sg, maxi = mx, degrad_size = 0)
    lingradG = mapkaG.sum(axis = 0)/mapkaG.shape[0]

    cm = dechannel(img_mins.copy(), 2)
    cp = dechannel(img_plus.copy(), 2)
    mapkaR, rangerR = advanced_map(cm, cp, sg = sg, maxi = mx, degrad_size = 0)
    lingradR = mapkaR.sum(axis = 0)/mapkaR.shape[0]
    
    return lingradB[cutoff:-cutoff], lingradG[cutoff:-cutoff], lingradR[cutoff:-cutoff], mapkaR


def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y


def get_channel_map(img_mins, img_plus, cutoff = 6, factor = 1, sg = 2, mx = 2):
    lingradB, lingradG, lingradR, mapkaR = get_bgr_channels(img_mins, img_plus, cutoff = cutoff, sg = sg, mx = mx)
    
    diffRG = lingradR-lingradG
    diffRB = lingradR-lingradB
    diff = (diffRG+diffRB)/2

    poly = list(np.polyfit(range(cutoff, len(diff)+cutoff), diff, 3))
    poly.reverse()
    arr_x = np.asarray(range(0, img_mins.shape[1]))
    fitDif = PolyCoefficients(arr_x, poly)
    
    final_map = mapkaR+fitDif*factor
    return final_map
  
    
def get_distribution(mapka):
    flat_dif = mapka.copy()
    flat_dif = flat_dif.flatten()
    flat_dif.sort()
    return flat_dif


def get_minmax(flat_dif, percent = 0.01):   
    count = flat_dif.shape[0]

    minval = flat_dif[int(count*percent)]
    maxval = flat_dif[-int(count*percent)]

    return minval, maxval


def get_amap_parameter(string, name):
    string = string[string.find(name)+1+len(name):]
    string = string[:string.find(',')]
    return string   


def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def ft_read(d, nbytes):
    s = d.read(nbytes)
    return s


def connect_to_elliptec(com_i, timeout = 3):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = timeout
    
    try:
        ser.open()    
        ser.write(b'0in')
        resp = ser.read(32)
        if resp == b'0IN0E114002352021150101680002300': 
            print('Connected to elliptec on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not elliptec')
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None


def connect_to_elliptec_top(com_i):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = 6
    
    try:
        ser.open()    
        ser.write(b'0in')
        resp = ser.read(32)
        if resp == b'0IN0E114008572021150101680002300': 
            print('Connected to elliptec on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not elliptec'%com_i)
            print('Response: %s'%resp)
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None


def connect_to_ell6k(com_i):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = 6
    
    try:
        ser.open()    
        ser.write(b'0in')
        resp = ser.read(32)
        if resp == b'0IN061060030620211201001F0000000': 
            print('Connected to elliptec2 on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not elliptec'%com_i)
            print(resp)
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None


def connect_to_thortemp(com_i):
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port='COM%d'%com_i
    ser.timeout = 6

    ser.parity = serial.PARITY_NONE
    ser.bytesize = serial.EIGHTBITS
    ser.stopbits = serial.STOPBITS_ONE
    
    try:
        ser.open()    
        ser.write(b'tact?\r')
        resp = ser.read_until(b'>')
        if str(resp).find('tact?') >= 0: 
            print('Connected to thortemp on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not thortemp'%com_i)
            print('Response: %s'%resp)
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None


def angle_to_ellocommand(value):
    value = int(value*398)
    if value < 0: 
        value_hex = str(hex(((abs(value) ^ 0xffffffff) + 1) & 0xffffffff))
    else:
        value_hex = str(hex(value))


    value_hex = value_hex[value_hex.find('x')+1:].zfill(8)
    value_hex = value_hex.replace('a', 'A')
    value_hex = value_hex.replace('b', 'B')
    value_hex = value_hex.replace('c', 'C')
    value_hex = value_hex.replace('d', 'D')
    value_hex = value_hex.replace('e', 'E')
    value_hex = value_hex.replace('f', 'F')

    return bytes('0ma%s'%value_hex.zfill(8), 'ascii')


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
    
    
def calib_get_postion(comp, sigma = 151):
    # check if it is not outside (no dot)
    if comp.max() < 25:
        return -1
    
    # blur and normalize
    comp = cv2.GaussianBlur(comp, (sigma, sigma), cv2.BORDER_DEFAULT)
    comp = (comp-comp.min())
    comp = comp/comp.max()*255
    
    # threshold, find countor, get center
    tresh = (comp.max()-comp.min())/2
    ret, thresh1 = cv2.threshold(np.uint8(comp), tresh, 1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    if M["m00"] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx = 0
        cy = 0
    
    return [cx, cy]


def num_to_coords(num, size = 10, dim = (1024, 768)):    
    nx = int(num%size)
    ny = int((num-num%size)/size)
    
    inc_x = int(dim[0]/size)
    inc_y = int(dim[1]/size)
    
    x = int(inc_x/2) + nx*inc_x
    y = int(inc_y/2) + ny*inc_y
    
#     print('x = %d, y = %d'%(x, y))
    return x, y


def get_homography_matrix():
    baseline = np.dot(np.load('calibration/baseline.npy')[...,:3], rgb_weights)
    # images = []
    # for i in range(0, calib_dots_dim**2):
        # temp = np.load('calibration/num%d.npy'%i)
        # images.append(np.dot(temp[...,:3], rgb_weights))
        
    coords_prj = []
    coords_cam = []
    sigma = 11

    combine = (baseline.copy()*0)
    for i in range(0, calib_dots_dim**2):
        img = np.load('calibration/num%d.npy'%i)
        img = np.dot(img[...,:3], rgb_weights)
        bimg = img-baseline
        combine += bimg/calib_dots_dim**2
        bimg = cv2.GaussianBlur(bimg, (sigma, sigma), cv2.BORDER_DEFAULT)
        x, y = np.where(bimg == bimg.max())
        c = [y[0], x[0]]
        coords_cam.append(c)
        crds = num_to_coords(i, size = calib_dots_dim)
        coords_prj.append([crds[0], crds[1]])
            
    coords_prj_arr = np.array(coords_prj)
    coords_cam_arr = np.array(coords_cam)
    
#     return coords_prj_arr, coords_cam_arr
    h, status = cv2.findHomography(coords_cam_arr, coords_prj_arr)
    im_out = cv2.warpPerspective(baseline, h, (1024, 768))
    
    fig = plt.figure(figsize = (12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.set_title('camera image')
    ax1.imshow(combine, cmap = 'gray')
    for i, crd in enumerate(coords_cam):
        ax1.text(crd[0], crd[1], str(i), c = 'red')
        
    ax2.set_title('projector array')
    ax2.imshow(im_out, cmap = 'gray')
    for i, crd in enumerate(coords_prj):
        ax2.text(crd[0], crd[1], str(i), c = 'white')
        
    fig.savefig('calibration/calib_result.png', dpi = 400)
    
    return h


def angle_abs_to_rel(ang_abs, offset):
    rel = ang_abs - offset
    if rel > 360: rel -= 360
    if rel < 0: rel += 360
    return rel


def angle_rel_to_abs(ang_rel, offset):
    absa = ang_rel + offset
    if absa > 360: absa -= 360
    if absa < 0: absa += 360
    return absa
    

def add_border(img, border, color = (0, 0, 0)):
    # initialize new, bigger image
    new_img = np.zeros((768+2*border, 1024+2*border, 3), np.uint8)
    new_img[:,:] = color
    
    # paste old img in the middle and resize to match original size
    new_img[border:768+border, border:1024+border] = img
    new_img = cv2.resize(new_img, (1024, 768))
    
    return new_img


def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb  


def connect_to_grbl(com_i, timeout = 3):
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port='COM%d'%com_i
    ser.timeout = timeout
    
    try:
        ser.open()    
        resp = ser.read(32)
        if resp == b"\r\nGrbl 0.9j ['$' for help]\r\n": 
            print('Connected to grbl on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not grbl')
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None
    
    
def send_to_grbl(ser, command, maxlen = 64):
    ser.write(command)
    full_response = b''
    resp = b''
    i = 0
    while (resp != b'ok\r\n') or (i > maxlen):
        resp = ser.readline()
        full_response += resp
        i += 1
        
    return full_response
    ser.write(command)
    full_response = b''
    resp = b''
    while (resp != b'ok\r\n'):
        resp = ser.readline()
        full_response += resp
        
    return full_response


def generate_random_image():
    fontpath = 'OpenSans-Regular.ttf'
    font11 = ImageFont.truetype(fontpath, 34)

    frame = np.zeros((448, 800, 3), np.uint8)

    words = 'Jamnik – jedna z ras psów pochodząca z Niemiec. Niemiecka nazwa jamnika Dachshund oznacza w dosłownym tłumaczeniu "borsuczy pies", etymologia nazwy związana jest z jego zbliżoną do borsuków budową oraz wykorzystywaniem tej rasy do polowania na zwierzęta ryjące nory.'.split(' ')
    # words = ['jamnik', 'chirality', 'nanoparticles', 'gold', 'liquid crystal', 'camera error', 'impact factor',
            # 'laser', 'samples', 'work', 'nematic', 'helical', 'danger', 'run', 'thorlabs', 'microscope', 'science',
            # 'strange', 'temperature']
    
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    for i in range(0, 40):
        clr = int(60 + 195*np.random.random())
        draw.text((int(700*np.random.random()), int(400*np.random.random())), np.random.choice(words), font = font11, fill = (clr, clr, clr))
    return np.array(img_pil)


def cmap_array(arr, cmap, rng = 1, middle = 0):
     return cmap((arr/rng+1-middle/rng)/2)[:,:,:3]


def domain_eater_out(point, size, eat_dir, eat_stage):
    # draw circle on empty array
    img = np.zeros((448, 800), np.uint8)
    img = cv2.circle(img, point, size, (255), -1)
    eat_dir -= 90
    eat = (np.sin(eat_dir*np.pi/180), np.cos(eat_dir*np.pi/180))
    
    # draw eater
#     eat_stage = 1-eat_stage
    if eat_stage < 0: eat_stage == 0
    ex = int(-2*size*eat_stage*eat[0] + point[0])
    ey = int(-2*size*eat_stage*eat[1] + point[1])
    eater = cv2.circle(np.zeros((448, 800), np.uint8), (ex, ey), size, (255), -1)
#     img = img - (img&(~eater))
    img = img&eater
    
    return cv2.merge((img, img, img))


def domain_eater_in(point, size, eat_dir, eat_stage):
    # draw circle on empty array
    img = np.zeros((448, 800), np.uint8)
    img = cv2.circle(img, point, size, (255), -1)
    eat_dir -= 90
    eat = (np.sin(eat_dir*np.pi/180), np.cos(eat_dir*np.pi/180))
    # draw eater moved at full size in opposite of eat direction - 
    
    if eat_stage < 0: eat_stage == 0
    eat_stage = 1-eat_stage
    ex = int((2*size*eat_stage)*eat[0] + point[0])
    ey = int((2*size*eat_stage)*eat[1] + point[1])
    eater = cv2.circle(np.zeros((448, 800), np.uint8), (ex, ey), size, (255), -1)
#     img = img - (img&(~eater))
    img = img-eater
    
    return cv2.merge((img, img, img))


def thortemp_check_enable(ser):
    if ser != None:
        ser.write(b'stat?\r')
        resp = ser.read_until(b'>')
        resp_str = str(resp)

        if resp_str.find('stat') >= 0:
            value = resp_str[resp_str.find('\\r')+2 : resp_str.find('>')-1]
            stat_byte = bin(int(value, 16))
            return int(stat_byte[-1])  # return first bit as it represents the state of ENABLE
    else:
        print('Not connected')
    
    
def thortemp_switch_enable(ser):
    if ser != None:
        ser.write(b'ens\r')
        
        
def thortemp_set_temp(ser, temp):
    if ser != None:
        ser.write(b'tset=%2.1f'%temp)
        
        
def thortemp_set_ramp(ser, ramp):
    if ser != None:
        ser.write(b'ramp=%d'%ramp)
        
        
def thortemp_get_temp(ser):
    if ser != None:
        ser.write(b'tact?\r')
        resp = ser.read_until(b'>')
        
        resp_str = str(resp)

        if resp_str.find('tact') >= 0:
            try:
                value = float(resp_str[resp_str.find('\\r')+2 : resp_str.find('C\\')-1])
                return value
            except:
                print('Some stupid error occured, but dont worry!')
                print(resp_str)
                return -1
    return -1


def move_time(cur_pos, pos, vel):
    dist = np.sqrt((pos[0]-cur_pos[0])**2 + (pos[1]-cur_pos[1])**2)
    return dist/(vel/60)
    
    
def focus_time(d, F):
    return abs(63*d/F)
    
    
def focus_check(pos_list, fcs_list):
    yvals = np.array(fcs_list)
    if yvals.mean()* 3 < yvals.max():
        return pos_list[yvals.argmax()]
    else:
        return None
     

def replace_variables(anim_str, variables):
    for n in range(0, len(variables)):
        var_name = 'var%d'%n
        anim_str = anim_str.replace(var_name, '%2.2f'%variables[n])            
    return anim_str


def draw_parametric_animation(anim_str, posx, posy, rota, size, ctime, variables = []):
    canvas = np.zeros((448, 800, 3), np.uint8)
    
    # scale parameter is from textfile, size is from brush size 
    anim_str = replace_variables(anim_str, variables)
    for obj in anim_str.split('OBJECT')[1:]:
        splitted = obj.splitlines()

        line1 = splitted[0].split(' ')
        typ = line1[1]

        # init parameters
        cx = 0
        cy = 0
        scale = 1

        for line in splitted[1:]:
            line_splt = line.split(' ')
#             line_splt = replace_variables(line_splt, variables)
            if eval(line_splt[0]) <= ctime:
                if eval(line_splt[2]) >= ctime:
                    total_time = eval(line_splt[2])-eval(line_splt[0])
                    completness = (ctime-eval(line_splt[0]))/total_time
                    if line_splt[3] == 'MOVE':
                        change_x = eval(line_splt[7])-eval(line_splt[4])
                        change_y = eval(line_splt[8])-eval(line_splt[5])
                        cx = int(eval(line_splt[4]) + change_x*completness)
                        cy = int(eval(line_splt[5]) + change_y*completness)
                        cx = int(cx*size/100 + 800/2)
                        cy = int(cy*size/100 + 448/2)
                    if line_splt[3] == 'SCALE':
                        change_s = eval(line_splt[6])-eval(line_splt[4])
                        scale = eval(line_splt[4]) + change_s*completness

        if line1[1] == 'rectangle':
            stp = (int(cx-0.5*eval(line1[2])*scale*size/100), int(cy-0.5*eval(line1[3])*scale*size/100))
            enp = (int(cx+0.5*eval(line1[2])*scale*size/100), int(cy+0.5*eval(line1[3])*scale*size/100))
            clr = (eval(line1[4]), eval(line1[4]), eval(line1[4]))
            canvas = cv2.rectangle(canvas, stp, enp, clr, -1)
            
        if line1[1] == 'ellipse':
            axes = (int(eval(line1[2])*scale*size/100), int(eval(line1[3])*scale*size/100))
            clr = (int(line1[5]), int(eval(line1[5])), int(eval(line1[5])))
            canvas = cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, clr, -1)

        # now rotate
        canvas_rot = imutils.rotate(canvas, rota)

        # now offset
        pil_img = Image.fromarray(canvas_rot)
        pil_img2 = ImageChops.offset(pil_img, int(posy-800/2), int(posx-448/2))
        canvas_done = np.array(pil_img2)

    return canvas_done


def get_animations():
    files = os.listdir('parametric_animations/')
    anims = [f.replace('.anim', '') for f in files if f.find('.anim') >= 0]
    return anims


def rnd_between(a, b):
    return b - (b-a)*random.random()


def pixelburn2_rnd_variables():
    v0 = rnd_between(0.1, 0.5)
    v1 = rnd_between(1, 1.5)
    v2 = rnd_between(0, 0.4)
    v3 = rnd_between(0.1, 0.8)
    v4 = 0
    while(v4 < v1):
        v4 = rnd_between(1, 1.8)
        
    return [v0, v1, v2, v3, v4]


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


def get_pattern_pixel(pattern, pixel):
    cols = pattern.shape[1]
    row = int((pixel-pixel%cols)/cols)
    col = int(pixel%cols)
    if pixel < (pattern.shape[0]*pattern.shape[1]):
        return pattern[row, col], row, col
    else:
        return -1, row, col


def make_coord_list(sqr_size = 460):
    cords = []
    for i in range(0, 5):
        x = 0.5*sqr_size
        y = (-0.5*sqr_size)*i/5
        cords.append([x, y])
        cords.append([-x, y])
        cords.append([-y, x])
        cords.append([y, -x])

        x = 0.5*sqr_size
        y = (0.5*sqr_size)*i/5
        cords.append([x, y])
        cords.append([-x, y])
        cords.append([-y, x])
        cords.append([y, -x])
    return cords


def make_var_list(cords):
    return [[60*v/(490/2) for v in c] for c in cords]


def make_side_mask(c, s = 100):
    width = s
    height = s

    maska = np.zeros((448, 800))
    cx = int(maska.shape[1]/2) + 30 + c[0]
    cy = int(maska.shape[0]/2) + c[1]
    maska = cv2.rectangle(maska, (int(cx-width/2), int(cy-height/2)), (int(cx+width/2), int(cy+height/2)), 1, -1)
    
    return maska


def select_side(mapka, target, check_size = 100, side_thresh = 0.8, thresh = 0.01):
    '''target = 1 or -1'''
    cords = make_coord_list()
    var_list = make_var_list(cords)
    posnegs = []
    for i in range(0, len(cords)):
        maska = make_side_mask(cords[i], s = check_size)
        posnegs.append(check_mapka_maske(mapka, maska, thresh = thresh))
        
    if target == 1:
        if max(posnegs) > side_thresh:
            ind = posnegs.index(max(posnegs))
            return var_list[ind]
        else:
            return [999, 999]
    if target == -1:
        if min(posnegs) < 1-side_thresh:
            ind = posnegs.index(min(posnegs))
            return var_list[ind]
        else:
            return [999, 999]
    
    
def check_mapka_maske(mapka, mask, thresh = 0.01):
    masked = mapka*mask
    
    pos = masked[masked > thresh].shape[0]
    neg = masked[masked < -thresh].shape[0]
    
    return pos/(pos+neg)


def should_try_kolo(anim_hist, gfs_hist):
    cur = len(anim_hist)-1
    if cur < 2: return False
    if anim_hist[cur] == 'pix' and anim_hist[cur-1] == 'pix' and anim_hist[cur-2] == 'pix':
        if gfs_hist[cur]-gfs_hist[cur-1] < 0.05 and gfs_hist[cur-1]-gfs_hist[cur-2] < 0.05:
            return True
    if len([v for v in anim_hist if v == 'pix']) > 14: return True
    return False


def translate_image(img, x, y):
    M = np.float32([
    [1, 0, x],
    [0, 1, y]
    ])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted


class CamReader_old(Thread):
    '''runs on separate thread and read image from camera'''
    def __init__(self, master):
        super().__init__()
        self.master = master

        self.cap = cv2.VideoCapture(0)
        self.frame = np.zeros((448, 800, 3), np.uint8)
        
        fontpath = 'OpenSans-Regular.ttf'
        self.font11 = ImageFont.truetype(fontpath, 34)
        self.frame_is_new = True
        
        self.change_gain(0)
        time.sleep(1)
        self.change_expo(-8)
        
        
    def read_frame(self):
        ret, self.frame = self.cap.read()
        # ret = False
        if ret == False:
            self.frame = generate_random_image()
        self.frame_is_new = True            
        # time.sleep(1)
        self.master.after(100, self.read_frame)
        
        
    def get_frame(self):
        if self.frame_is_new == True:
            self.frame_is_new = False
            return self.frame, True
        else:
            return self.frame, False
            
    
    def change_gain(self, gain):
        parameter = cv2.CAP_PROP_GAIN
        self.cap.set(parameter, gain)
        print('new camera gain %s'%str(gain))
        
    
    def change_expo(self, expo):
        parameter = cv2.CAP_PROP_EXPOSURE
        self.cap.set(parameter, expo)
        print('new camera expo %s'%str(expo))
        
        
    def get_frame_shape(self):
        return self.frame.shape


class CamReader_old_but_not_very_old(Process):
#     override the constructor
    def __init__(self, event, sharr, config_arr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.config_arr = config_arr
        self.gain = self.config_arr[0]
        self.expo = self.config_arr[1]
        self.sharr = sharr
        self.data = Value('i', 0)
        # self.cap = cv2.VideoCapture(0)
        self.frame = np.zeros(img_shape, np.uint8)

    # override the run function
    def run(self):
        self.cap = cv2.VideoCapture(0)
        while(True):       
            # read frame from cap object (if fail generate random img)
            ret, frame = self.cap.read()
            if ret == False:
                frame = generate_random_image()
                
            frame = cv2.resize(frame, (800, 448))
                
            # copy frame to shared array (sharr)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*img_shape)
            np.copyto(self.sharr_np, frame)
            
            
            # check if config requires change
            # if self.gain != self.config_arr[0]:
                # self.change_gain(self.config_arr[0])
            # if self.expo != self.config_arr[1]:
                # self.change_expo(self.config_arr[1])
            
            time.sleep(0.075)

            
            
            if self.event.is_set():
                break
            
    
    def change_gain(self, gain):
        parameter = cv2.CAP_PROP_GAIN
        self.cap.set(parameter, gain)
        print('new camera gain %s'%str(gain))
        
    
    def change_expo(self, expo):
        parameter = cv2.CAP_PROP_EXPOSURE
        self.cap.set(parameter, expo)
        print('new camera expo %s'%str(expo))


class XBOX(Process):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)
    def __init__(self, xbox_event):
        Process.__init__(self)
        # initialize integer attribute
        self.xbox_event = xbox_event
        
        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        print('Pad controller initialized')


    def read(self): # return the buttons/triggers that you care about in this methode
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.X # b=1, x=2
        rb = self.RightBumper
        rtr = self.RightTrigger
        return [x, y, rb, rtr, a, b]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XBOX.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XBOX.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XBOX.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XBOX.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XBOX.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XBOX.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.X = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state
        
        
class CamReader(Process):
#     override the constructor
    def __init__(self, event, sharr, config_arr, camparam_arr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.config_arr = config_arr
        self.camparam_arr = camparam_arr
        self.gain = self.config_arr[0]
        self.expo = self.config_arr[1]
        self.expoAbs = self.config_arr[2]
        self.sharr = sharr
        self.data = Value('i', 0)
        # self.cap = cv2.VideoCapture(0)
        self.frame = np.zeros(img_shape, np.uint8)
        self.change_expo_thisloop = 0
        
        self.hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.width = ueye.INT()
        self.height = ueye.INT()
        print('cam reader initialized')
        
        
    def init_ui_camera(self):      
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
            
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
            
        
        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")


        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
                
        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = self.ueye.IS_CM_MONO8
            self.nBitsPerPixel = self.ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")
            
            
        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)


        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("New camera loaded")
            
            
        # FPS fix
        time.sleep(2)
        
        print('Attemtping FPS change')
        
        
        fps = ueye.double()
        ret = ueye.is_GetFramesPerSecond(self.hCam, fps)
        print('Current FPS', ret, fps)
        time.sleep(0.5)
        
        number = ueye.UINT()
        ret = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, ueye.UINT(90), ueye.sizeof(number))
        print('PxCLK set:',ret, number)
        ret = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET, number, ueye.sizeof(number))
        print('PxCLK current:',ret, number)
        time.sleep(0.5)
        
        fps = ueye.double(15)
        fps_actual = ueye.double()
        ret = ueye.is_SetFrameRate (self.hCam, fps, fps_actual)
        print('Changing FPS', ret, fps, fps_actual)
        time.sleep(1)
        
        fps = ueye.double()
        ret = ueye.is_GetFramesPerSecond(self.hCam, fps)
        print('Current FPS', ret, fps)
        time.sleep(0.5)
        


    # override the run function
    def run(self):
        self.init_ui_camera()
        while(True):
            try:
                # read frame from new camera
                array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
                # bytes_per_pixel = int(nBitsPerPixel / 8)
                # ...reshape it in an numpy array...
                frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
                # ...resize the image by a half
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                
                # ret, frame = self.cap.read()
                # if ret == False:
                    # frame = generate_random_image()
                    
                # frame = cv2.resize(frame, (800, 448))
                
                # correctly resize frame
                w, h = frame.shape[1], frame.shape[0]
                frame = cv2.resize(frame, (int(w*(448/800)), 448))
                final_img = np.zeros((448, 800, 4), dtype = np.ubyte)
                pos_x = int((800-frame.shape[1])/2)
                final_img[:, pos_x:pos_x + frame.shape[1], :] = frame
                if self.camparam_arr[2] == 1:
                    final_img[:,:,2] = final_img[:,:,0]
            except:
                final_img = generate_random_image()
                
            # copy frame to shared array (sharr)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*img_shape)
            np.copyto(self.sharr_np, final_img[:,:,:3])
            
            
            # check if config requires change
            # if self.gain != self.config_arr[0]:
                # self.change_gain(self.config_arr[0])
            # if self.expo != self.config_arr[1]:
                # self.change_expo(self.config_arr[1])
            
            time.sleep(0.010)
            
            if self.config_arr[1] != 0:
                time.sleep(0.01)
                self.pParam = ueye.double()
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, self.pParam, 8)
                self.pParam.value = self.pParam.value + self.config_arr[1]
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self.pParam, 8)
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, self.pParam, 8)
                self.camparam_arr[1] = int(self.pParam.value)
                print('cur exp read:', self.pParam.value)
                self.config_arr[1] = 0
                    
            if self.config_arr[2] != -1:
                time.sleep(0.01)
                self.pParam = ueye.double()
                self.pParam.value = self.config_arr[2]
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self.pParam, 8)
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, self.pParam, 8)
                self.camparam_arr[1] = int(self.pParam.value)
                print('cur exp read:', self.pParam.value)
                self.config_arr[2] = -1
                
                
            # time.sleep(0.275)
            # self.pParam = ueye.double()
            # self.pParam.value = 30
            # ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self.pParam, 8)
            # time.sleep(0.275)

            if self.event.is_set():
                print('Cam reader: Call to exit camera received!')
                # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
                ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

                # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
                ueye.is_ExitCamera(self.hCam)
                break
            
    
    def change_gain(self, gain):
        parameter = cv2.CAP_PROP_GAIN
        self.cap.set(parameter, gain)
        print('new camera gain %s'%str(gain))
        
    
    def change_expo(self, expo):
        print('change expo is', self.change_expo_thisloop)
        self.change_expo_thisloop = -1
        print('change expo is', self.change_expo_thisloop)

    
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        
        # loadable internal variables
        self.ell_offset_bottom = 0  # this was previously called abs_ang
        self.ell_offset_top = 0
        self.ell_offset_bottom = 0
        self.elli_bottom_abs = 0
        self.elli_top_abs = 0
        self.pulseOn = 1
        self.pulseOff = 1
        self.border = 90  # border added to projector image to counter the fact, that camera does not see full projected laser image
        self.loop_delay_target = 20  # it will try to match this delay (todo: actully implement this)
        
        # non-loadable variables (are modified inside app)
        self.interaction_mode = 'none'
        self.erosion = 0
        self.erosion_counter = 0
        self.erosion_delay = 1
        self.hold_proj = 0   
        self.camera_overlay = 0
        self.brush_size = 50
        self.always_overlay = 0
        self.camera_gain = 0
        self.camera_should_change_gain = False
        self.camera_expo = -8
        self.camera_should_change_expo = False
        self.camera_image_type = 'RAW'
        self.camera_map_counter = 0
        self.frame_map_switch = 20
        self.polrot = 10
        self.show_parameters = 1  # determines whether to add information on camera image before display (scale, temp and stuff)
        self.current_obiektyw = 'X10'
        self.loop_cnt = 0  # loop counter, for displayng loop informations
        self.loop_time = 0
        self.time_prev = 0
        self.save_data = False  # if True saves raw data during show_frame
        self.rec_data = False  # if True saves raw frame with specified freqency
        self.rec_data_cnt = 0  # used to count the save freq during show_frame
        self.prev_grayframe = np.zeros((448, 800), float)  # this stores camera frame as grayscale for differential polarization map
        self.prev_frame  = np.zeros((448, 800, 3), np.uint8)
        self.diff_cmap = plt.get_cmap('PiYG')
        self.calib_grad = np.load('calibration_gradient.npy')
        self.calib_factor = 1.2
        self.map_rng = 0.01
        self.map_gauss = 0
        self.cv2image_const = np.zeros((448, 800, 3), np.uint8)  # this is for 'MAP'
        self.pt_eat_dir = 0  # for point laser: eat directin
        # self.pt_eat_speed = 3  # for point laser: how much eat_factor increases every mainloop
        # self.pt_size = 100  # how big is circle made by point
        self.pt_eat_stage = 0  # current eat stage
        self.pt_point = (400, 500)
        self.camera_point_draw = np.zeros((448, 800, 3), np.uint8)  # this is space where points are drawn
        self.spol_angle = 0  # used for save frames at different polarizations during SPOL
        self.spol_date = None  # as above
        self.atp_cnt = 0  # main autopilot counter
        self.pos_list = []
        self.timer_wait_secs = 0  # used to synchronize breaks during automatic actions
        self.timer_set_at = datetime.datetime.now()  # as above
        self.atp_display_map = False
        self.advanced_mapka = np.zeros((448, 800), float)
        self.amap_ranger = 0.6
        self.mnloop_timer_started = 0  # for counting time within mainloop
        self.position_list = []
        self.cpos = 0
        self.aclear = False
        self.shot_cnt = 0
        self.illuminate_pattern = True
        self.blinker_multip = 1
        self.mnloop_total_tenths = 0
        self.ramp_on = False
        self.focus = 0
        self.afocus_flag = False
        self.afocus_step = -0.5
        self.afocus_dir = 1
        self.fcs = None
        img = Image.open('patterns/chess.png')
        arr = np.array(img)
        self.current_pattern = arr[:,:,0]/255
        self.pixel_current = 0
        self.kolo_direction = 0
        self.pixelburns_in_row = 0
        
        # parametric animation
        self.paranim_start_time = None
        self.paranim_x = 0
        self.paranim_y = 0
        self.paranim_angle = 0
        self.paranim_length_ms = 3000
        self.paranim_str = ''
        self.paranim_variables = []
        self.camera_lo_gain = 0
        self.camera_hi_gain = 50
        self.extvars = {}
        self.pixelburn_count = 0
        self.burn_hist_gfs = []
        self.burn_hist_anims = []
        self.burn_params_comment = ''
        self.frame_lapse = 1
        self.frame_lapse_counter = 0
        self.brightness = 0
        self.bright_cnt = 0
        self.offsets = np.load(rmd + 'rotation_compensation/offsets.npy')
        
        # laser stuff
        self.laserduty = 0.5
        self.laserstate = 'OFF'
        self.rigol = None
        
        # ell6k
        self.ell6k = None
        self.ell6_state = 'OUT'
        
        # thortemp
        self.thortemp = None
        self.thortemp_Tactual = -1
        
        # mm-map
        self.mm_counter = 0
        self.mm_count = 3
        
        # self.cam_reader = CamReader(self.master)
        # self.cam_reader.read_frame()
        
        
        self.event = Event()
        self.sharr = RawArray('c', 448*800*3)  # holds camera frame
        self.config_arr = RawArray('i', (6, 8, -1))  # hold the camera configuration (gain, expo)
        self.camparam_arr = RawArray('i', (0, 0, 0))  # hold the camera configuration (gain, expo, invert_colors)
        self.cam_reader = CamReader(self.event, self.sharr, self.config_arr, self.camparam_arr)
        
        
        self.xbox_event = Event()
        self.xbox_controller = None
        
        # start the process
        self.cam_reader.start()
        
        
        self.elliptec = None
        self.elliptec_top = None
        self.video_writer = None
        self.projector_window = None
        self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        self.grbl = None
        self.grblX = 0.0
        self.grblY = 0.0
        self.grblZ = 0.0
        self.B1_was_pressed = False
        self.thrm = None
        self.t_cur = -1  # current temperature (set by refresh_temperature) OBSOLETE?
        self.rot_around = 90
        
        self.frame10counter = 0
        
        # self.projector_arr = cv2.circle(self.projector_arr, (500, 300), 50, (255, 255, 255), -1)
        
        # raw animation varaibles
        self.raw_projector_screen = False
        self.raw_proj_imgs = []
        self.raw_proj_counter = 0
        self.raw_proj_curr_img = np.zeros((768, 1024, 3), np.uint8)
       
        fontpath = 'OpenSans-Regular.ttf'
        self.font11 = ImageFont.truetype(fontpath, 11)
        
        # this is grayscale image that overlays the camera and it is projected through homography to projector screen
        self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.camera_draw = cv2.circle(self.camera_draw, (500, 300), 50, (255, 255, 255), -1)
        
        # matrix for homography (is set during calibration)
        self.homomatrix = np.zeros((3,3))        
        self.image_baseline = np.zeros((600, 800, 3), np.uint8)
        
        # logo for fun/testing
        self.logo = cv2.imread('idle_patterns/unilogo.png')
        self.rownosc = cv2.imread('idle_patterns/rownosc.png')
        self.default_image = self.logo
        self.projector_arr = self.default_image
        self.pattern_image = cv2.imread('pattern_images/pattern_unilogo.png')
        
        # this is mouse xy relative to camera canvas
        self.mouse_x = 0
        self.mouse_y = 0

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.main_loop()
        self.elli_refresh('COM1')
        self.read_extvars()
    
    
    #Creation of init_window
    def init_window(self):
        col = 0
        row = 0
        self.save_image = False

        # changing the title of our master widget      
        self.master.title("Mikroskop control")
        
        # menu
        menu = Menu(root)
        self.master.config(menu=menu)
        
        
        fileMenu = Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="Reload variables", command=self.read_extvars)
        fileMenu.add_command(label = 'Load image', command = self.load_image)
        fileMenu.add_command(label = 'Load pattern', command = self.load_patterning_pattern)
        fileMenu.add_command(label = 'Connect all subsystems', command = self.connect_all)
        fileMenu.add_command(label = 'Connect pad', command = self.connect_pad)
        
        
        # submenu
        idleMenu = Menu(root, tearoff=False)
        fileMenu.add_cascade(label="Idle pattern", menu=idleMenu)
        idleMenu.add_command(label = 'Logo', command = lambda: self.change_default_image(self.logo))
        idleMenu.add_command(label = 'Black', command = lambda: self.change_default_image(self.rownosc))
        
        fileMenu.add_command(label="Exit", command=self.exit)
        
        objectiveMenu = Menu(root, tearoff=False)
        menu.add_cascade(label="Change objective", menu=objectiveMenu)
        objectiveMenu.add_command(label = 'X5', command = lambda: self.change_objective('X5'))
        objectiveMenu.add_command(label = 'X10', command = lambda: self.change_objective('X10'))
        objectiveMenu.add_command(label = 'X20', command = lambda: self.change_objective('X20'))
        objectiveMenu.add_command(label = 'X50', command = lambda: self.change_objective('X50'))
        objectiveMenu.add_command(label = 'X100', command = lambda: self.change_objective('X100'))
        
        specialMenu = Menu(root, tearoff=False)
        menu.add_cascade(label="Advanced programs", menu=specialMenu)
        specialMenu.add_command(label="Run projector calibration", command=self.start_calib)
        specialMenu.add_command(label="Load projector calibration", command=self.load_calib)
        specialMenu.add_command(label="Tune parameters", command = self.autotune_anim_params)
        specialMenu.add_command(label="Single-tune", command = self.autotune_single)
        specialMenu.add_command(label="Paterning", command = self.start_patterning)
        specialMenu.add_command(label="Illumination program", command = self.ill_program)
        specialMenu.add_command(label="Get offsets (obsolete?)", command = self.autop_find_offset)
        specialMenu.add_command(label="Cross polarizers (todo)", command=self.autop_crosspolarize)
        
        
        # keybindings
        self.master.bind('[', self.brush_dec)
        self.master.bind(']', self.brush_inc)
        self.master.bind('<Escape>', self.esc_key_btn)
        self.master.bind('<Shift-Key-C>', self.key_shiftc)
        self.master.protocol("WM_DELETE_WINDOW", self.exit)
        
        
        #  -- FIRST COLUMN --
        fcol_frame = Frame(root)
        fcol_frame.grid(row=0, column=0, padx = padd)
        
        self.canvas = Canvas(fcol_frame, width=800-4, height=600, bg='black')
        # self.canvas.grid(row=row, column=col, padx = padd)
        self.canvas.pack(fill = Y, padx = padd)
        # self.canvas.bind("<Double-Button-1>", self.image_move)  # for future interactions        
        self.canvas.bind("<ButtonPress-1>", self.mouse_motionB1)
        # self.canvas.bind("<ButtonRelease-1>", self.cam_btn_release)
        self.canvas.bind('<B1-Motion>', self.mouse_motionB1)
        self.canvas.bind('<Motion>', self.mouse_motion)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_B1_release)
        # row += 1
        
        # CAMERA CONTROLS FRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        self.label = Label(current_frame, text = 'Camera controls: ')
        self.label.pack(side =  LEFT)
        
        # self.label = Label(current_frame, text = 'Gain: ')
        # self.label.pack(side =  LEFT)
        
        # self.btnGdg = Button(current_frame, text = '-5', command = lambda: self.camera_deltagain(-5))
        # self.btnGdg.pack(side = LEFT)
        # self.btnGdg = Button(current_frame, text = '-1', command = lambda: self.camera_deltagain(-1))
        # self.btnGdg.pack(side = LEFT)
        
        # self.labelGain = Label(current_frame, text = '%d'%self.camera_gain)
        # self.labelGain.pack(side =  LEFT)
        
        # self.btnGdg = Button(current_frame, text = '+1', command = lambda: self.camera_deltagain(1))
        # self.btnGdg.pack(side = LEFT)
        # self.btnGdg = Button(current_frame, text = '+5', command = lambda: self.camera_deltagain(5))
        # self.btnGdg.pack(side = LEFT)
        
        
        self.btnInvCol = Button(current_frame, text = 'Invert colors', command = lambda: self.invert_colors())
        self.btnInvCol.pack(side = LEFT)
        
        
        self.label= Label(current_frame, text = '   Exposure:')
        self.label.pack(side =  LEFT)
        
        self.btnGdg = Button(current_frame, text = '-25', command = lambda: self.camera_deltaexpo(-25))
        self.btnGdg.pack(side = LEFT)
        
        self.btnGdg = Button(current_frame, text = '-5', command = lambda: self.camera_deltaexpo(-5))
        self.btnGdg.pack(side = LEFT)
        
        self.labelExpo = Label(current_frame, text = '%d'%self.camera_expo)
        self.labelExpo.pack(side =  LEFT)
        
        self.btnGdg = Button(current_frame, text = '+5', command = lambda: self.camera_deltaexpo(+5))
        self.btnGdg.pack(side = LEFT)        
        
        self.btnGdg = Button(current_frame, text = '+25', command = lambda: self.camera_deltaexpo(+25))
        self.btnGdg.pack(side = LEFT)
        
        
        
        # IMAGE CONTROLS FRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        self.label = Label(current_frame, text = 'Display controls: ')
        self.label.pack(side =  LEFT)
        
        # source_var = StringVar(self.master)
        # source_var.set(0) # default value
        # self.source_menu = OptionMenu(current_frame, source_var, *list_cameras(), command = self.mod_source)
        # self.source_menu.pack(side = LEFT)
        
        self.btnHoldProj = Button(current_frame, text = 'Hold', command = self.btn_hold_proj)
        self.btnHoldProj.pack(side = LEFT)
        self.btnAlwaysOverlay = Button(current_frame, text = 'Always overlay', command = self.btn_always_overlay)
        self.btnAlwaysOverlay.pack(side = LEFT)
        
        # DRAWING CONTROLS FRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        self.label = Label(current_frame, text = 'Camera draw controls: ')
        self.label.pack(side =  LEFT)
        
        self.btnDraw = Button(current_frame, text = 'Draw', command = self.btn_draw)
        self.btnDraw.pack(side = LEFT)      
        self.btnPoint = Button(current_frame, text = 'Point', command = self.btn_point)
        self.btnPoint.pack(side = LEFT)             
        self.btnErosion = Button(current_frame, text = 'Erosion', command = self.btn_erosion)
        self.btnErosion.pack(side = LEFT)          
        self.btnAnim = Button(current_frame, text = 'Animation', command = self.btn_anim)
        self.btnAnim.pack(side = LEFT)
        self.btnImg = Button(current_frame, text = 'Image', command = self.btn_image)
        self.btnImg.pack(side = LEFT) 
        
        eros_var = StringVar(self.master)
        eros_var.set(1) # default value
        self.mnErospeed = OptionMenu(current_frame, eros_var, *[1, 2, 3, 4, 5, 8, 10, 15, 20], command = self.mn_set_erospeed)
        self.mnErospeed.pack(side = LEFT)
        
        self.btnClear = Button(current_frame, text = 'Clear', command = self.btn_clear_camdraw)
        self.btnClear.pack(side = LEFT)
        
        # IMAGE TYPE FRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        self.label = Label(current_frame, text = 'Image type controls: ')
        self.label.pack(side =  LEFT)
        
        self.btnIcntRaw = Button(current_frame, text = 'Raw', command = self.btn_icnt_raw)
        self.btnIcntRaw.pack(side = LEFT)        
        self.btnIcntRaw.config(relief=SUNKEN)
        self.btnIcntDyn = Button(current_frame, text = 'Dynamic', command = self.btn_icnt_dyn)
        self.btnIcntDyn.pack(side = LEFT)
        self.btnIcntMap = Button(current_frame, text = 'Map', command = self.btn_icnt_map)
        self.btnIcntMap.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Blink speed: ')
        self.label.pack(side =  LEFT)
        fms_var = StringVar(self.master)
        fms_var.set(40) # default value
        self.mnFMS = OptionMenu(current_frame, fms_var, *[20, 25, 30, 40, 50, 60, 80, 100, 200], command = self.mn_set_fms)
        self.mnFMS.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Rotation: ')
        self.label.pack(side =  LEFT)
        polrot_var = StringVar(self.master)
        polrot_var.set(10) # default value
        self.mnPolrot = OptionMenu(current_frame, polrot_var, *[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15], command = self.mn_set_polrot)
        self.mnPolrot.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'around:')
        self.label.pack(side =  LEFT)
        self.rotaround_var = StringVar(self.master)
        self.rotaround_var.set('90') # default value
        self.rotaEntered = Entry(current_frame, width = 4, textvariable = self.rotaround_var)
        self.rotaEntered.pack(side = LEFT) 
        
        
        # IMAGE TYPE FRAME 2 (ADVANCED MAPPING)
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        self.label = Label(current_frame, text = 'Advanced mapping controls:')
        self.label.pack(side =  LEFT)
        
        self.btnAdvMap = Button(current_frame, text = 'Degradient map', command = self.btn_adv_map)
        self.btnAdvMap.pack(side = LEFT)
        
        self.btnAdvMap = Button(current_frame, text = 'R-channel map', command = self.btn_rchan_map)
        self.btnAdvMap.pack(side = LEFT)
        
        self.advMapOptions = StringVar()
        self.advMapOptions.set('G=2, M=3, CH=-1, B=1')
        self.nameEntered = Entry(current_frame, width = 30, textvariable = self.advMapOptions)
        self.nameEntered.pack(side = LEFT)        
        
        self.btnAdvMap = Button(current_frame, text = 'Save pol frames', command = self.btn_save_polframes)
        self.btnAdvMap.pack(side = LEFT)
        
        # self.label = Label(current_frame, text = 'G - gauss sigma, M - maxfilter sigma, Df - degradient resize factor, Tf - domian threshold factor')
        # self.label.pack(side =  LEFT)
        

        # SAVING CONTROLS FRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # current_frame.config(bg = "#"+("%06x"%random.randint(0,16777215)))
        # row += 1
        
        self.label_save = Label(current_frame, text = 'Sample name: ')
        self.label_save.pack(side =  LEFT)
        
        self.path = StringVar()
        self.nameEntered = Entry(current_frame, width = 15, textvariable = self.path)
        self.nameEntered.pack(side = LEFT)
        
        self.buttonSave = Button(current_frame, text = 'Save', command = self.save_img)
        self.buttonSave.pack(side = LEFT)
        
        self.buttonRecord = Button(current_frame, text = 'Record video', command = self.record_video)
        self.buttonRecord.pack(side = LEFT)
        
        self.buttonRecord2 = Button(current_frame, text = 'Record timelapse', command = self.record_timelapse)
        self.buttonRecord2.pack(side = LEFT)
        
        self.buttonsaveData = Button(current_frame, text = 'Save frame', command = self.save_data_button)
        self.buttonsaveData.pack(side = LEFT)
        
        self.buttonrecData = Button(current_frame, text = 'Record frames', command = self.rec_data_button)
        self.buttonrecData.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Freq: ')
        self.label.pack(side =  LEFT)
        self.savedata_freq_var = StringVar(self.master)
        self.savedata_freq_var.set(10) # default value
        self.mnSDfreq = OptionMenu(current_frame, self.savedata_freq_var, *[1, 2, 5, 8, 10, 20, 50, 100])
        self.mnSDfreq.pack(side = LEFT)
        
        # LOOP TIME INFO
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # row += 1
        
        self.label_looptime = Label(current_frame, text = 'Loop time: 0.0000')
        self.label_looptime.pack(side =  LEFT)
        
        # autop_frame SUBFRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # row += 1
        self.create_autop_frame(current_frame)
        
        # areal controls SUBFRAME
        current_frame = Frame(fcol_frame)
        current_frame.pack(fill = Y, padx = padd)
        # current_frame.grid(row=row, column=col, padx = padd)
        # row += 1
        self.create_areal_frame(current_frame)

        
        #  -- SECOND COLUMN --
        row = 0
        col = 1
        
        # control_frame (main frame that holds multiple control frames for elliptec, g-code and others)
        control_frame = Frame(root)
        control_frame.grid(row=row, column=col, padx = padd)
        row += 1
        ctrl_pad = 4
        
        self.lab = Label(control_frame, text = 'SUBSYSTEMS CONTROLS')
        self.lab.config(font=group_name_font)
        # self.lab.grid(row=0, column=0, padx = ctrl_pad, pady = ctrl_pad)
        self.lab.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        
        
        # ELLIPTEC TOP SUBFRAME
        eli_frame = Frame(control_frame)
        # eli_frame.grid(row=1, column=0, padx = ctrl_pad, pady = ctrl_pad)
        eli_frame.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_elliptec_frame(eli_frame)
        
        # ELLIPTEC BOTTOM SUBFRAME
        eli_frame2 = Frame(control_frame)
        # eli_frame2.grid(row=1, column=0, padx = ctrl_pad, pady = ctrl_pad)
        eli_frame2.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_elliptec_frame2(eli_frame2)
        
        # GRBL SUBFRAME
        grbl_frame = Frame(control_frame)
        # grbl_frame.grid(row=2, column=0, padx = ctrl_pad, pady = ctrl_pad)
        grbl_frame.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_grbl_frame(grbl_frame)
        
        # THERMAL SUBFRAME
        thermal_frame = Frame(control_frame)
        # thermal_frame.grid(row=3, column=0, padx = ctrl_pad, pady = ctrl_pad)
        thermal_frame.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_thermal_frame(thermal_frame)
        
        # RIGOL SUBFRAME
        rigol_frame = Frame(control_frame)
        # rigol_frame.grid(row=4, column=0, padx = ctrl_pad, pady = ctrl_pad)
        rigol_frame.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_rigol_frame(rigol_frame)        
        
        # RIGOL SUBFRAME
        ell6_frame = Frame(control_frame)
        # ell6_frame.grid(row=5, column=0, padx = ctrl_pad, pady = ctrl_pad)
        ell6_frame.pack(fill = Y, padx = ctrl_pad, pady = ctrl_pad)
        self.create_ell6_frame(ell6_frame)
        
        
        
        #  -- THIRD COLUMN --
        row = 0
        col = 2
        
        # projector control frame
        proj_frame = Frame(root)
        proj_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        proj_frame1 = Frame(proj_frame)
        proj_frame1.grid(row=0, column=0, padx = padd)
        
        self.init_proj_win_btn = Button(proj_frame1, text = 'Init projector window', command = self.initiate_projector_window)
        self.init_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(proj_frame1, text = 'Activate projector window', command = self.activate_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(proj_frame1, text = 'Close projector window', command = self.close_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        # for raw projector output controls
        proj_frame2 = Frame(proj_frame)
        proj_frame2.grid(row=1, column=0, padx = padd)
        
        self.label_raw = Label(proj_frame2, text = 'Raw img controls: ')
        self.label_raw.pack(side =  LEFT)
        
        self.btn_raw_enable = Button(proj_frame2, text = 'Enable', command = self.raw_enable)
        self.btn_raw_enable.pack(side = LEFT)
        
        self.raw_load_b = Button(proj_frame2, text = 'Load seq', command = self.btn_raw_load)
        self.raw_load_b.pack(side = LEFT)
        
        self.raw_load_b = Button(proj_frame2, text = 'Play seq', command = self.btn_raw_play)
        self.raw_load_b.pack(side = LEFT)
        
        self.label_raw = Label(proj_frame2, text = 'Frame del [ms]: ')
        self.label_raw.pack(side =  LEFT)
        
        self.rawdelay_var = StringVar()
        self.rawdelay_var.set('300')
        self.rawdelay_entry = Entry(proj_frame2, width = 8, textvariable = self.rawdelay_var)
        self.rawdelay_entry.pack(side = LEFT, fill = X)
        
        
        
        canvas_frame = Frame(proj_frame)
        canvas_frame.grid(row=2, column=0, padx = padd)
        
        self.proj_mirror_canvas = Canvas(canvas_frame, width=256, height=192, bg='black')
        self.proj_mirror_canvas.pack(side = LEFT)
        
        # console
        console_frame = Frame(proj_frame)
        console_frame.grid(row=3, column=0, padx = padd)
        
        self.console_box = Text(console_frame, height = 30, width = 80, background = '#00112b', foreground = '#c5d3e8', font=("Consolas", 8))
        self.console_box.pack(fill = Y, padx = padd, side = LEFT)
        
        scroll_bar = Scrollbar(console_frame, command=self.console_box.yview)
        scroll_bar.pack(side=RIGHT, expand=True, fill = Y)
        self.console_box.configure(yscrollcommand=scroll_bar.set)
        
        long_text = """Welcome to mikroskop control!
"""
        
        self.console_box.insert(END, long_text)
    
    
    def btn_raw_load(self):        
        self.raw_proj_imgs = filedialog.askopenfilenames()
        self.log(self.raw_proj_imgs)
        
        
    def btn_raw_play(self):
        delay = int(self.rawdelay_var.get())
        self.raw_proj_counter = 0
        self.master.after(delay, self.raw_output_next_frame)
        self.log('Raw output animation started')
        
    
    def raw_output_next_frame(self):
        if self.raw_proj_counter < len(self.raw_proj_imgs):
            self.raw_proj_curr_img = cv2.imread(self.raw_proj_imgs[self.raw_proj_counter])
            self.raw_proj_curr_img = cv2.resize(self.raw_proj_curr_img, (1024, 768), interpolation = cv2.INTER_AREA)
            self.raw_proj_counter += 1
            self.log('Raw output animation frame %d displayed'%self.raw_proj_counter)
            delay = int(self.rawdelay_var.get())
            self.master.after(delay, self.raw_output_next_frame)
        else:
            self.log('Raw output animation ended at frame %d'%self.raw_proj_counter)
        
    
    def connect_pad(self):
        if self.xbox_controller == None:        
            self.xbox_controller = XBOX(self.xbox_event)
    

    def invert_colors(self):
        invert = 0
        if self.camparam_arr[2] == 1 : invert = 0
        if self.camparam_arr[2] == 0 : invert = 1
        self.camparam_arr[2] = invert
    
    
    def log(self, text, should_print = True, color = None, bold = None):
        numlines = int(self.console_box.index('end - 1 line').split('.')[0])
        
        self.console_box['state'] = 'normal'
        if numlines >= 256:
            self.console_box.delete(1.0, 2.0)
        if self.console_box.index('end-1c')!='1.0':
            self.console_box.insert('end', '\n')
        self.console_box.insert('end', text)
        self.console_box.see('%3.1f'%(numlines+1))
        self.console_box['state'] = 'disabled'
        
        if should_print == True:
            print(text)
    
        
    def create_elliptec_frame(self, eli_frame):       
        # elliptec frame name
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell = Label(cur_frame, text = 'ELL14 TOP CONTROL')
        self.label_ell.config(font=('Segoe UI', 14, 'bold'))
        self.label_ell.pack(side =  LEFT)
        
        # elliptec position frames
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell_top= Label(cur_frame, text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(0, 0, 0))
        self.label_ell_top.config(font=('Segoe UI', 13))
        self.label_ell_top.pack(side =  LEFT)
                
        
        # elliptec connection frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        self.ell_com_var = StringVar(self.master)
        self.ell_com_var.set('COM3') # default value
        self.ell_com_menu = OptionMenu(cur_frame, self.ell_com_var, *serial_ports(), command = self.elli_refresh_top)
        self.ell_com_menu.pack(side = LEFT)
        
        self.label_ell_status = Label(cur_frame, text = 'Elliptec status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.label_ell_status2_top = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_ell_status2_top.pack(side =  LEFT)       
        

        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell1 = Label(cur_frame, text = 'Rotate relative: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var2_top = StringVar()
        self.ell_var2_top.set(str(angle_abs_to_rel(0, self.ell_offset_top)))
        self.nameEntered2 = Entry(cur_frame, width = 15, textvariable = self.ell_var2_top)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate2 = Button(cur_frame, text = 'Rotate', command = self.rotate_elli_rel_top)
        self.buttonRotate2.pack(side = LEFT)        
        self.button = Button(cur_frame, text = 'Set 90 here', command = self.elli_top_set_offset)
        self.button.pack(side = LEFT)
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y) 
        self.buttonRotateStepM1 = Button(cur_frame, text = '-10', command = lambda: self.rotate_elli_step_top(-10))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-5', command = lambda: self.rotate_elli_step_top(-5))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-1', command = lambda: self.rotate_elli_step_top(-1))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.25', command = lambda: self.rotate_elli_step_top(-0.25))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.05', command = lambda: self.rotate_elli_step_top(-0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '+0.05', command = lambda: self.rotate_elli_step_top(0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+0.25', command = lambda: self.rotate_elli_step_top(0.25))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+1', command = lambda: self.rotate_elli_step_top(1))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+5', command = lambda: self.rotate_elli_step_top(5))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+10', command = lambda: self.rotate_elli_step_top(10))
        self.buttonRotateStepP1.pack(side = LEFT)
        
        
    def create_elliptec_frame2(self, eli_frame):       
        # elliptec frame name
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell = Label(cur_frame, text = 'ELL14 BOTTOM CONTROL')
        self.label_ell.config(font=('Segoe UI', 14, 'bold'))
        self.label_ell.pack(side =  LEFT)
        
        # elliptec position frames
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell_bottom = Label(cur_frame, text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(0, 0, 0))
        self.label_ell_bottom.config(font=('Segoe UI', 13))
        self.label_ell_bottom.pack(side =  LEFT)
                
        
        # elliptec connection frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        ell_com_var2 = StringVar(self.master)
        ell_com_var2.set('COM7') # default value
        self.ell_com_menu = OptionMenu(cur_frame, ell_com_var2, *serial_ports(), command = self.elli_refresh)
        self.ell_com_menu.pack(side = LEFT)
        
        self.label_ell_status = Label(cur_frame, text = 'Elliptec status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.label_ell_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_ell_status2.pack(side =  LEFT)       
        

        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell1 = Label(cur_frame, text = 'Rotate relative: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var2 = StringVar()
        self.ell_var2.set(str(angle_abs_to_rel(0, self.ell_offset_bottom)))
        self.nameEntered2 = Entry(cur_frame, width = 15, textvariable = self.ell_var2)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate2 = Button(cur_frame, text = 'Rotate', command = self.rotate_elli_rel)
        self.buttonRotate2.pack(side = LEFT)        
        self.button = Button(cur_frame, text = 'Set 90 here', command = self.elli_bottom_set_offset)
        self.button.pack(side = LEFT)
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y) 
        self.buttonRotateStepM1 = Button(cur_frame, text = '-10', command = lambda: self.rotate_elli_step(-10))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-5', command = lambda: self.rotate_elli_step(-5))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-1', command = lambda: self.rotate_elli_step(-1))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.25', command = lambda: self.rotate_elli_step(-0.25))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.05', command = lambda: self.rotate_elli_step(-0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '+0.05', command = lambda: self.rotate_elli_step(0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+0.25', command = lambda: self.rotate_elli_step(0.25))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+1', command = lambda: self.rotate_elli_step(1))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+5', command = lambda: self.rotate_elli_step(5))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+10', command = lambda: self.rotate_elli_step(10))
        self.buttonRotateStepP1.pack(side = LEFT)
    

    def create_ell6_frame(self, eli_frame):       
        # elliptec frame name
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.labeeli6 = Label(cur_frame, text = 'ELLIPTEC 6 CONTROL')
        self.labeeli6.config(font=subsystem_name_font)
        self.labeeli6.pack(side =  LEFT)
        
        # elliptec position frames
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.lab_ell6_state = Label(cur_frame, text = 'ELL6 state = %s'%self.ell6_state)
        self.lab_ell6_state.config(font=('Segoe UI', 13))
        self.lab_ell6_state.pack(side =  LEFT)
                
        # elliptec connection frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        ell_com_var = StringVar(self.master)
        ell_com_var.set('COM5') # default value
        self.ell_com_menu = OptionMenu(cur_frame, ell_com_var, 'COM8', *serial_ports(), command = self.elli6_refresh)
        self.ell_com_menu.pack(side = LEFT)
        
        self.label_ell6_status = Label(cur_frame, text = 'Elliptec 6 status: ')
        self.label_ell6_status.pack(side =  LEFT)
        
        self.label_ell6_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_ell6_status2.pack(side =  LEFT)

        # button in out frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        self.btnOut = Button(cur_frame, text = 'Mirror OUT', command = self.btnAct_ell6_out)
        self.btnOut.pack(side = LEFT)
        self.btnIn = Button(cur_frame, text = 'Mirror IN', command = self.btnAct_ell6_in)
        self.btnIn.pack(side = LEFT)
       

    def create_grbl_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab = Label(cur_frame, text = 'GRBL CONTROL')
        self.lab.config(font=subsystem_name_font)
        self.lab.pack(side =  LEFT)
        
        # grbl position frames
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_grblpos = Label(cur_frame, text = 'X = %2.4f, Y = %2.4f'%(0,0))
        self.lab_grblpos.config(font=('Segoe UI', 13))
        self.lab_grblpos.pack(side =  LEFT)
        
        
        # grbl set/move connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        # set frame
        set_frame = Frame(cur_frame)
        set_frame.grid(row=0, column=0)
        
        sze_frame = Frame(set_frame)
        sze_frame.grid(row=0, column=0)     
        lab = Label(sze_frame, text = 'Step size [mm]:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.grbl_size = StringVar()
        self.grbl_size.set('0.2')
        self.grbl_size_entry = Entry(sze_frame, width = 15, textvariable = self.grbl_size)
        self.grbl_size_entry.pack(side = LEFT, fill = X)
        
        spd_frame = Frame(set_frame)
        spd_frame.grid(row=1, column=0)         
        lab = Label(spd_frame, text = 'Step speed [mm/min]:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)    
        self.grbl_speed = StringVar()
        self.grbl_speed.set('10.0')
        self.grbl_speed_entry = Entry(spd_frame, width = 15, textvariable = self.grbl_speed)
        self.grbl_speed_entry.pack(side = LEFT, fill = X)
        
        btn_frame = Frame(set_frame)
        btn_frame.grid(row=2, column=0)
        btn = Button(btn_frame, text = 'Set params')
        btn.pack(side = LEFT)
        btn = Button(btn_frame, text = 'Reset zero', command = lambda: self.grbl_command('G10 P0 L20 X0 Y0 Z0\n'))
        btn.pack(side = LEFT)
        
        # move frame
        mov_frame = Frame(cur_frame)
        mov_frame.grid(row=0, column=1)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=0, column=0)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'U', command = lambda: self.grbl_command('G21 G91 G1 Y-%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=1, column=0)
        btn = Button(bframe, text = 'L', command = lambda: self.grbl_command('G21 G91 G1 X-%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '.', command = self.grbl_reset_zero)
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'R', command = lambda: self.grbl_command('G21 G91 G1 X%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=2, column=0)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'D', command = lambda: self.grbl_command('G21 G91 G1 Y%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        
        #z axis frame (focus control)
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
               
        lab = Label(cur_frame, text = 'Z-Axis [µm]:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side = LEFT)
        
        btn = Button(cur_frame, text = '-100', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(-0.1)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = '-10', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(-0.01)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = '-2', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(-0.002)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = '+2', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(+0.002)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = '+10', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(0.01)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = '+100', command = lambda: self.grbl_command('G21 G91 G1 Z%s F10\n'%str(0.1)))
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = 'Auto', command = self.btn_autofocus)
        btn.pack(side = LEFT)
        btn = Button(cur_frame, text = 'Fine', command = self.btn_autofocus_fine)
        btn.pack(side = LEFT)
        
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        grbl_com_var = StringVar(self.master)
        grbl_com_var.set('COM4') # default value
        self.grbl_com_menu = OptionMenu(cur_frame, grbl_com_var, *serial_ports(), command = self.grbl_refresh)
        self.grbl_com_menu.pack(side = LEFT)
        
        self.label_grbl_status = Label(cur_frame, text = 'GRBL status: ')
        self.label_grbl_status.pack(side =  LEFT)
        
        self.label_grbl_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_grbl_status2.pack(side =  LEFT) 


    def create_thermal_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab = Label(cur_frame, text = 'THERMAL CONTROL')
        self.lab.config(font=subsystem_name_font)
        self.lab.pack(side =  LEFT)
        
        # current temp
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_tempcur = Label(cur_frame, text = 'T = %3.1f °C'%self.thortemp_Tactual)
        self.lab_tempcur.config(font=('Segoe UI', 13))
        self.lab_tempcur.pack(side =  LEFT)
        
        
        # set temp frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
    
        lab = Label(cur_frame, text = 'Set temperature:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.set_temp_var = StringVar()
        self.set_temp_var.set('40')
        self.set_temp_entry = Entry(cur_frame, width = 15, textvariable = self.set_temp_var)
        self.set_temp_entry.pack(side = LEFT, fill = X)
        
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.set_temperature)
        self.set_temp_btn.pack(side = LEFT)
        
        # ramp frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        lab = Label(cur_frame, text = 'Ramp rate:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.set_ramp_var = StringVar()
        self.set_ramp_var.set('5')
        self.set_ramp_entry = Entry(cur_frame, width = 5, textvariable = self.set_ramp_var)
        self.set_ramp_entry.pack(side = LEFT, fill = X)
        
        lab = Label(cur_frame, text = 'target:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.set_ramp_target_var = StringVar()
        self.set_ramp_target_var.set('40')
        self.set_ramp_target_entry = Entry(cur_frame, width = 5, textvariable = self.set_ramp_target_var)
        self.set_ramp_target_entry.pack(side = LEFT, fill = X)
        
        self.set_ramp_btn = Button(cur_frame, text = 'Ramp!', command = self.ramp_clicked)
        self.set_ramp_btn.pack(side = LEFT)
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        grbl_com_var = StringVar(self.master)
        grbl_com_var.set('COM6') # default value
        self.grbl_com_menu = OptionMenu(cur_frame, grbl_com_var, *serial_ports(), command = self.thrm_refresh)
        self.grbl_com_menu.pack(side = LEFT)
        
        self.label_thrm_status = Label(cur_frame, text = 'THERMAL status: ')
        self.label_thrm_status.pack(side =  LEFT)
        
        self.label_thrm_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_thrm_status2.pack(side =  LEFT) 
        

    def create_rigol_frame(self, frame):
        # rigol frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.labLaser = Label(cur_frame, text = 'LASER CONTROL')
        self.labLaser.config(font = subsystem_name_font)
        self.labLaser.pack(side =  LEFT)
        
        # laser status
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_laser = Label(cur_frame, text = 'DUTY = %2.2f %%, LASER IS %s'%(0.0, 'OFF'), fg = laser_off_color)
        self.lab_laser.config(font=('Segoe UI', 13))
        self.lab_laser.pack(side =  LEFT)
        
        # controller laser variable max power
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.label = Label(cur_frame, text = 'Pad max variable laser power:')
        self.label.pack(side =  LEFT)
        
        # laserduty_var = StringVar(self.master)
        # laserduty_var.set(0.5) # default value
        # self.mnPolrot = OptionMenu(cur_frame, laserduty_var, *[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99], command = self.mn_set_laserduty)
        # self.mnPolrot.pack(side = LEFT)
        
        self.laserduty_var_pad = StringVar()
        self.laserduty_var_pad.set('10')
        self.laserduty_entry_pad = Entry(cur_frame, width = 15, textvariable = self.laserduty_var_pad)
        self.laserduty_entry_pad.pack(side = LEFT, fill = X)
        
        
        # laser duty control
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.label = Label(cur_frame, text = 'Set duty cycle:')
        self.label.pack(side =  LEFT)
        
        # laserduty_var = StringVar(self.master)
        # laserduty_var.set(0.5) # default value
        # self.mnPolrot = OptionMenu(cur_frame, laserduty_var, *[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99], command = self.mn_set_laserduty)
        # self.mnPolrot.pack(side = LEFT)
        
        self.laserduty_var = StringVar()
        self.laserduty_var.set('40')
        self.laserduty_entry = Entry(cur_frame, width = 15, textvariable = self.laserduty_var)
        self.laserduty_entry.pack(side = LEFT, fill = X)
        
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.btn_laserduty)
        self.set_temp_btn.pack(side = LEFT)
        
        
        # laser state control
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btnLaserState = Button(cur_frame, text = 'Turn on the laser', command = self.btn_laser_switch, width = 24)
        self.btnLaserState.pack(side =  LEFT)
        
        # connection to rigol
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btn_rigol_connect = Button(cur_frame, text = 'Connect to Rigol', command = self.btn_rigolcon)
        self.btn_rigol_connect.pack(side = LEFT)
        
        self.label_rigol_status = Label(cur_frame, text = 'RIGOL status: ')
        self.label_rigol_status.pack(side =  LEFT)
        
        self.label_rigol_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_rigol_status2.pack(side =  LEFT) 
     
    
    def create_autop_frame(self, frame):
        # autop frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab = Label(cur_frame, text = 'AUTOPILOT CONTROL')
        self.lab.config(font=group_name_font)
        self.lab.pack(side =  LEFT)
        
        # simple button for now
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btnAutop = Button(cur_frame, text = 'Areal map', command = self.autop_areal_map)
        self.btnAutop.pack(side = LEFT)
        
        self.labpm = Label(cur_frame, text = 'Polars (+/-):')
        self.labpm.pack(side = LEFT)    
        self.areal_pols_var = StringVar()
        self.areal_pols_var.set('5')
        self.apols_entry = Entry(cur_frame, width = 12, textvariable = self.areal_pols_var)
        self.apols_entry.pack(side = LEFT) 
                
        self.btnBMap = Button(cur_frame, text = 'mm-MAP', command = self.btn_mmap)
        self.btnBMap.pack(side = LEFT)
        
        self.labrots = Label(cur_frame, text = 'Rotations:')
        self.labrots.pack(side = LEFT)    
        self.map_rots_var = StringVar()
        self.map_rots_var.set('0')
        self.map_rots_entry = Entry(cur_frame, width = 12, textvariable = self.map_rots_var)
        self.map_rots_entry.pack(side = LEFT) 
        
        self.mm_var = StringVar(self.master)
        self.mm_var.set(5) # default value
        self.omenu_mm = OptionMenu(cur_frame, self.mm_var, *[2, 3, 5, 8, 10, 15, 20, 30, 50])
        self.omenu_mm.pack(side = LEFT)
        
        self.eatdir_var = StringVar(self.master)
        self.eatdir_var.set(1) # default value
        self.mnED = OptionMenu(cur_frame, self.eatdir_var, *[-1, 1])
        self.mnED.pack(side = LEFT)
        
        self.btnAClear = Button(cur_frame, text = 'Area clear!', command = self.btn_area_clear)
        self.btnAClear.pack(side = LEFT)
        
        self.btnPat = Button(cur_frame, text = 'Pattern array', command = self.btn_pattern_array2)
        self.btnPat.pack(side = LEFT)
        
        self.label = Label(cur_frame, text = 'Ill. time [s]: ')
        self.label.pack(side =  LEFT)
        
        self.set_ill_var = StringVar()
        self.set_ill_var.set('3')
        self.set_ill_entry = Entry(cur_frame, width = 4, textvariable = self.set_ill_var)
        self.set_ill_entry.pack(side = LEFT, fill = X)
        
        # moved from temporary controls
        current_frame = Frame(frame)
        current_frame.pack(fill = Y)

        self.label = Label(current_frame, text = 'Animation: ')
        self.label.pack(side =  LEFT)
        self.ptact_var = StringVar(self.master)
        self.ptact_var.set('--- select ---') # default value
        self.mnSDfreq = OptionMenu(current_frame, self.ptact_var, *get_animations())
        self.mnSDfreq.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Anim time [s]: ')
        self.label.pack(side =  LEFT)
        self.eatspeed_var = StringVar(self.master)
        self.eatspeed_var.set(5) # default value
        self.mnSDfreq = OptionMenu(current_frame, self.eatspeed_var, *eatspeed_value_list)
        self.mnSDfreq.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Anim vars: ')
        self.label.pack(side =  LEFT)
        
        self.paranim_var_str = StringVar()
        self.paranim_var_str.set('v0=0.2, v1=1.55, v2=0.3, v3=0.75, v4=1.45')
        self.paranim_entry = Entry(current_frame, width = 40, textvariable = self.paranim_var_str)
        self.paranim_entry.pack(side = LEFT) 
        
        
    def create_areal_frame(self, frame):
        # autop frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab = Label(cur_frame, text = 'AREAL CONTROL')
        self.lab.config(font=group_name_font)
        self.lab.pack(side =  LEFT)
        
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y) 
        self.label = Label(cur_frame, text = 'X step:')
        self.label.pack(side =  LEFT)
        self.arealc_x = StringVar()
        self.arealc_x.set('0.5')
        self.arealc_x_entry = Entry(cur_frame, width = 5, textvariable = self.arealc_x)
        self.arealc_x_entry.pack(side = LEFT)
        
        self.label = Label(cur_frame, text = 'Y step:')
        self.label.pack(side =  LEFT)
        self.arealc_y = StringVar()
        self.arealc_y.set('0.5')
        self.arealc_y_entry = Entry(cur_frame, width = 5, textvariable = self.arealc_y)
        self.arealc_y_entry.pack(side = LEFT)
        
        self.label = Label(cur_frame, text = 'Step count X:')
        self.label.pack(side =  LEFT)
        self.arealc_stepsX = StringVar()
        self.arealc_stepsX.set('3')
        self.arealc_steps_entryX = Entry(cur_frame, width = 5, textvariable = self.arealc_stepsX)
        self.arealc_steps_entryX.pack(side = LEFT)
        
        self.label = Label(cur_frame, text = 'Y :')
        self.label.pack(side =  LEFT)
        self.arealc_stepsY = StringVar()
        self.arealc_stepsY.set('3')
        self.arealc_steps_entryY = Entry(cur_frame, width = 5, textvariable = self.arealc_stepsY)
        self.arealc_steps_entryY.pack(side = LEFT)
    
    
    def connect_all(self):
        self.elli_refresh_top('COM3')
        self.elli_refresh('COM7')
        self.grbl_refresh('COM4')
        self.thrm_refresh('COM6')
        self.btn_rigolcon()
        self.elli6_refresh('COM5')
        
        
    def load_patterning_pattern(self):
        filename = filedialog.askopenfilename()
        img = Image.open(filename)
        arr = np.array(img)
        self.current_pattern = arr[:,:,0]/255
        
    
    def main_loop(self):  
        global projector_calib_c       
        self.show_frame()
        
        if self.projector_window != None:
            # refresh the image on actual projector screen
            self.refresh_projector_image()
            
            # display scaled copy of image in the main window
            resized = cv2.resize(self.projector_arr, (256, 192), interpolation = cv2.INTER_AREA)
            if self.hold_proj == 1:
                resized = cv2.putText(resized, 'PROJECTOR BLANKED', (12, 24), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            
            img = Image.fromarray(resized)
            self.imgproj_scaled = ImageTk.PhotoImage(image=img)
            self.proj_mirror_canvas.create_image(0, 0, image=self.imgproj_scaled, anchor=NW)
            
            
            # camera draw erosion
            if self.erosion == 1 and self.hold_proj != 1:
                if self.camera_draw.max() != 0:
                    if self.erosion_counter == self.erosion_delay:
                        self.camera_draw = cv2.erode(self.camera_draw, ckernel)
                        self.erosion_counter = 0
                    self.erosion_counter += 1
                    
            
        # absolute time-based action (per 0.1 second)
        self.mnloop_time_passed = time.time() - self.mnloop_timer_started
        if self.mnloop_time_passed >= 0.1:
                
            if self.interaction_mode == 'image' and self.hold_proj != 1:
                self.camera_point_draw = cv2.resize(self.pattern_image, (800, 448), interpolation = cv2.INTER_AREA)
                # self.camera_point_draw = cv2.resize(self.pattern_image, (1024, 768), interpolation = cv2.INTER_AREA)
                
                
            if self.interaction_mode == 'parametric_animation' and self.hold_proj != 1 and self.paranim_start_time != None:
                time_elapsed_ms = 1000*(time.time()-self.paranim_start_time)
                paranim_stage = time_elapsed_ms/self.paranim_length_ms
                if paranim_stage <= 1:
                    self.camera_point_draw = draw_parametric_animation(self.paranim_str, self.paranim_x, self.paranim_y, self.paranim_angle, self.brush_size, paranim_stage, variables = self.paranim_variables)
                else: 
                    self.paranim_start_time = None
                    self.camera_point_draw *= 0  # blank when finish
                    
                    
            # time-interval based actions
            if self.mnloop_total_tenths%50 == 0:
                if self.ramp_on == True:
                    temp_var = float(self.set_temp_var.get())
                    ramp_var = float(self.set_ramp_var.get())
                    ramp_target = float(self.set_ramp_target_var.get())
                    
                    if ramp_target == temp_var:
                        temp_var += 0.01
                        
                    ramp_var *= (ramp_target-temp_var)/abs(ramp_target-temp_var)  # get the sign of ramp                    
                    temp_var += ramp_var/60*5
                    if (ramp_var >= 0 and temp_var >= ramp_target) or (ramp_var <= 0 and temp_var <= ramp_target):
                        temp_var = ramp_target
                        self.ramp_clicked()
                        
                    self.set_temp_var.set('%2.2f'%temp_var)
                    self.set_temperature()
                    
            self.mnloop_total_tenths += 1
            if self.mnloop_time_passed > 2: 
                self.mnloop_time_passed = 2
                # self.log('Timer is behind, catching up!')
                
            self.mnloop_timer_started = time.time()-self.mnloop_time_passed+0.1
            
        
        if projector_calib_c < calib_dots_dim**2:
            self.proj_calib()
            
            # if projector_calib_c == 0:
                # self.homomatrix = get_homography_matrix()
        
        # timing stuff
        self.loop_cnt += 1
        time_now = time.time()
        self.loop_time += time_now-self.time_prev
        self.time_prev = time_now
        
        if self.loop_cnt >= 10:
            self.loop_cnt = 0            
            self.label_looptime.config(text = 'Main loop time: %f ms (%f Hz), ldt = %d'%(self.loop_time/10*1000, 1/(self.loop_time/10), self.loop_delay_target))
            if self.loop_time/10 < 0.1:
                self.label_looptime.config(fg='green')
            else:
                self.label_looptime.config(fg='red')
            
            self.loop_time = 0
        
        self.master.after(self.loop_delay_target, self.main_loop)
        
        
    def ramp_clicked(self):
        if self.thortemp != None:
            if self.ramp_on == True:
                set_ramp = 0
            if self.ramp_on == False:
                set_ramp = 1
                
            if set_ramp == 0:
                self.ramp_on = False
                self.set_ramp_btn.config(relief=RAISED, bg = '#f0f0f0')
                self.log('Ramp finished!')
                
            if set_ramp == 1:
                self.ramp_on = True
                self.set_ramp_btn.config(relief=SUNKEN, bg = '#ffe1b5')
                self.log('Initated ramp!')
        else:
            log('Cannot ramp withouth the connection to the thermal control device!')
        
        
    def get_paranim_vars(self):
        lis = self.paranim_var_str.get().split(', ')
        
        try:
            variables = []
            for name in lis:
                value = float(name[3:])
                variables.append(value)
            return variables
        except:
            self.log('Error in paranim variables string!')
            return []


    def btn_area_clear(self):
        self.aclear = True


    def btn_laserduty(self):
        self.laserduty = float(self.laserduty_var.get())
        self.mn_set_laserduty(self.laserduty)


    def camera_deltagain(self, deltagain):
        # self.camera_should_change_gain = True
        self.camera_gain += deltagain
        if self.camera_gain > 63 : self.camera_gain = 63
        if self.camera_gain < 0 : self.camera_gain = 0
        # self.labelGain.config(text = '%d'%self.camera_gain)


    def camera_deltaexpo(self, deltexpo):        
        self.config_arr[1] = deltexpo
        time.sleep(0.1)
        
        self.master.after(750, self.update_camparams_text)
        
        
    def camera_setexpo(self, expo):        
        self.config_arr[2] = expo
        time.sleep(0.1)
        
        self.master.after(750, self.update_camparams_text)
       
 
    def update_camparams_text(self):
        self.labelExpo.config(text = '%d'%self.camparam_arr[1])
        
        
    def grbl_get_position(self):
        if self.grbl != None:
            resp = send_to_grbl(self.grbl, b'?\n')
            self.grblX, self.grblY, self.grblZ = [float(f) for f in resp[resp.find('WPos:')+5:resp.find('>\\r\\nok')].split(',')]
            self.lab_grblpos.config(text = 'X = %2.4f, Y = %2.4f'%(self.grblX, self.grblY))
            
            
    def grbl_command(self, command):
        # print(command)
        if self.grbl != None:
            # ask for current position
            resp = str(send_to_grbl(self.grbl, b'?\n'))
            self.grblX, self.grblY, self.grblZ = [float(f) for f in resp[resp.find('WPos:')+5:resp.find('>\\r\\nok')].split(',')]
            
            if resp.find('Idle') >= 0:                
            # recalculate new position based on on direction
                step_size = float(self.grbl_size.get())
                
                if command.find('-') >= 0:
                    sgn = -1
                else:
                    sgn = 1
                    
                if command.find('X') >= 0: self.grblX += sgn * step_size
                if command.find('Y') >= 0: self.grblY += sgn * step_size
                
                send_to_grbl(self.grbl, bytes(command, encoding = 'utf-8'))
                # self.grbl.write(bytes('G4 P0\n', encoding = 'utf-8'))
                # self.grbl.write(bytes(command, encoding = 'utf-8'))
                
            self.lab_grblpos.config(text = 'X = %2.4f, Y = %2.4f'%(self.grblX, self.grblY))
    
    
    def ask_grbl_position(self):
        resp = str(send_to_grbl(self.grbl, b'?\n'))
        self.grblX, self.grblY, self.grblZ = [float(f) for f in resp[resp.find('WPos:')+5:resp.find('>\\r\\nok')].split(',')]
        self.lab_grblpos.config(text = 'X = %2.4f, Y = %2.4f'%(self.grblX, self.grblY))
    
    
    def grbl_reset_zero(self):
         resp = str(send_to_grbl(self.grbl, b'G10 P0 L20 X0 Y0 Z0\n'))
         # resp = str(send_to_grbl(self.grbl, b'G10 P1 L20 X0 Y0 Z0\n'))
         time.sleep(0.5)
         self.ask_grbl_position()
         
    
    def change_objective(self, obj):
        self.current_obiektyw = obj
     
        
    def read_extvars(self):
        f = open('external_variables.txt', 'r')
        cnt = f.read().split('\n')
        f.close()

        self.extvars = {}
        for i in range(0, len(cnt)):
            cnt[i] = cnt[i].replace(' ', '')
            comment = cnt[i].find('#')
            if comment >= 0:
                cnt[i] = cnt[i][:comment]

            if len(cnt[i]) > 0:
                name, val = cnt[i].split(':')
                self.extvars[name] = val
                
        # set the variables
        # self.pulseOff = extvars['pulseOff']
        # self.pulseOn = extvars['pulseOn']
        self.border = int(self.extvars['border'])
        self.loop_delay_target = int(self.extvars['loop_delay_target'])
        self.calib_factor = float(self.extvars['calib_factor'])
        self.map_rng = float(self.extvars['map_rng'])
        self.map_gauss = int(self.extvars['map_gauss'])
        self.ell_offset_bottom = float(self.extvars['ell_bottom_offset'])
        self.camera_lo_gain = int(self.extvars['camera_lo_gain'])
        self.camera_hi_gain = int(self.extvars['camera_hi_gain'])
        
        # offsets
        self.offsets = np.load(rmd + 'rotation_compensation/offsets.npy')
        
    
    def change_default_image(self, img):
        self.default_image = img
    
    
    def load_image(self):
        filename = filedialog.askopenfilename()
        self.pattern_image = cv2.imread(filename)
    
    
    def load_image_path(self, path):
        self.pattern_image = cv2.imread('c:/Users/Laser/Documents/mikroskop_control/pattern_images/' + path)
        

    def mn_set_polrot(self, value):
        '''this sets the polarization rotation angle for autorotation'''
        self.polrot = int(value)
        self.camera_map_counter = 0
        
        
    def mn_set_laserduty(self, value):
        '''this changes self.laserduty variable, also updates the label'''
        self.laserduty = float(value)
        self.lab_laser.config(text = 'DUTY = %2.2f %%, LASER IS %s'%(self.laserduty, self.laserstate))
        
        if self.rigol != None:
            self.rigol.write(':SOUR1:FUNC:SQU:DCYC %2.2f'%self.laserduty)
            time.sleep(0.25)
            resp = self.rigol.query(':SOUR1:FUNC:SQU:DCYC?')
            self.log('Laser duty cycle set to %s.'%resp.replace('\n', ''))
    

    def mn_set_fms(self, value):
        '''this sets the polarization rotation frequency for autorotation'''
        self.frame_map_switch = int(value)
        self.camera_map_counter = 0
    

    def btn_icnt_raw(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntRaw.config(relief=SUNKEN)
        self.camera_image_type = 'RAW'


    def btn_icnt_dyn(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntDyn.config(relief=SUNKEN)
        self.camera_image_type = 'DYN'


    def btn_icnt_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntMap.config(relief=SUNKEN)
        self.camera_image_type = 'MAP'
    
    
    def btn_adv_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_image_type = 'AMAP'
        self.camera_map_counter = 0
        self.esc_key_btn(1)
    
    
    def btn_rchan_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_image_type = 'AMAP2'
        self.camera_map_counter = 0
        self.esc_key_btn(1)
        
        
    def btn_save_polframes(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_image_type = 'SPOL'
        self.camera_map_counter = 0
        self.esc_key_btn(1)
    
    
    def brush_dec(self, event):
        if self.brush_size > 10:
            self.brush_size -= 10
            self.log('Brush size %d'%self.brush_size)
  
  
    def brush_inc(self, event):
        if self.brush_size < 300:
            self.brush_size += 10
            self.log('Brush size %d'%self.brush_size)
    
    
    def mn_set_erospeed(self, value):
        self.erosion_delay = int(value)
        self.erosion_counter = 0
    
 
    def btn_draw(self):
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        self.btnAnim.config(relief=RAISED)
        self.btnImg.config(relief=RAISED)
        if self.interaction_mode == 'draw':
            to_set = 'none'
            self.btnDraw.config(relief=RAISED)
        if self.interaction_mode != 'draw':
            to_set = 'draw'            
            self.btnDraw.config(relief=SUNKEN)
        self.interaction_mode = to_set
        
        
    def btn_point(self):
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        self.btnAnim.config(relief=RAISED)
        self.btnImg.config(relief=RAISED)
        if self.interaction_mode == 'parametric_animation':
            to_set = 'none'
            self.btnPoint.config(relief=RAISED)
        if self.interaction_mode != 'parametric_animation':
            to_set = 'parametric_animation'            
            self.btnPoint.config(relief=SUNKEN)
        self.interaction_mode = to_set
        
        
    def btn_anim(self):
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        self.btnAnim.config(relief=RAISED)
        self.btnImg.config(relief=RAISED)
        if self.interaction_mode == 'animation':
            to_set = 'none'
            self.btnAnim.config(relief=RAISED)
        if self.interaction_mode != 'animation':
            to_set = 'animation'            
            self.btnAnim.config(relief=SUNKEN)
        self.interaction_mode = to_set

        
    def btn_image(self):
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        self.btnAnim.config(relief=RAISED)
        self.btnImg.config(relief=RAISED)
        if self.interaction_mode == 'image':
            to_set = 'none'
            self.btnImg.config(relief=RAISED)
        if self.interaction_mode != 'image':
            to_set = 'image'            
            self.btnImg.config(relief=SUNKEN)
        self.interaction_mode = to_set
        
        
    def btn_hold_proj(self):
        to_set = 0
        if self.hold_proj == 1:
            to_set = 0
            self.btnHoldProj.config(relief=RAISED)
            self.camera_overlay = 0
        if self.hold_proj == 0:
            to_set = 1            
            self.btnHoldProj.config(relief=SUNKEN)
            self.camera_overlay = 0.3
        self.hold_proj = to_set
        
        
    def btn_erosion(self):
        to_set = 0
        if self.erosion == 1:
            to_set = 0
            self.btnErosion.config(relief=RAISED)
        if self.erosion == 0:
            to_set = 1            
            self.btnErosion.config(relief=SUNKEN)
        self.erosion = to_set
        
        
    def btn_laser_switch(self):
        if self.rigol != None:
            # switch laser state
            if self.laserstate == 'OFF': to_set = 'ON'
            if self.laserstate == 'ON': to_set = 'OFF'
            self.laserstate = to_set
            
            # update button state
            if self.laserstate == 'ON':
                self.btnLaserState.config(relief=SUNKEN, bg = laser_on_color)
                self.rigol.write(':OUTP1 ON')
                clr = laser_on_color
            if self.laserstate == 'OFF':   
                self.btnLaserState.config(relief=RAISED, bg = '#f0f0f0')
                self.rigol.write(':OUTP1 OFF')
                clr = laser_off_color
                
            # update label
            self.lab_laser.config(text = 'DUTY = %2.2f %%, LASER IS %s'%(self.laserduty, self.laserstate), fg=clr)
        else:
            messagebox.showwarning(title='Laser', message='Connection to Rigol is not established. To power up laser, connect to Rigol first')

        
    def btn_always_overlay(self):
        to_set = 0
        if self.always_overlay == 1:
            to_set = 0
            self.btnAlwaysOverlay.config(relief=RAISED)
        if self.always_overlay == 0:
            to_set = 1            
            self.btnAlwaysOverlay.config(relief=SUNKEN)
        self.always_overlay = to_set
        
        
    def btn_clear_camdraw(self):
        self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.pt_eat_stage = 1  # also clears point patterns
        self.paranim_start_time = None
        self.camera_point_draw *= 0
    
    
    def key_shiftc(self, event):
        self.btn_clear_camdraw()
    
 
    def mouse_motionB1(self, event):
        x = event.x
        y = event.y - 76 + 50  # to account for bigger canvas than camera image
        
        if self.interaction_mode == 'draw':
            self.draw_oncamera(x, y)
            
        if self.interaction_mode == 'point' or self.interaction_mode == 'parametric_animation':
            if self.B1_was_pressed == False:
                self.B1_pressed_at = (x, y)
                self.B1_was_pressed = True
                
        # if self.interaction_mode == 'parametric_animation':
            # if self.B1_was_pressed == False:
                # self.B1_pressed_at = (x, y)
                # self.B1_was_pressed = True
            
            
        # sets the mouse coords for drawing brush circle
        self.mouse_x = event.x
        self.mouse_y = event.y


    def mouse_motion(self, event):
        '''
        (only when the button is not pressed)
        '''
        self.mouse_x = event.x
        self.mouse_y = event.y
    
    
    def mouse_B1_release(self, event):
        if self.interaction_mode == 'point' and self.B1_was_pressed == True:
            self.create_point(self.B1_pressed_at[0], self.B1_pressed_at[1])
            x = event.x - self.B1_pressed_at[0]
            y = -(event.y - self.B1_pressed_at[1])
            self.pt_eat_dir = np.arctan(y/(x+0.00001))*180/np.pi
            if self.pt_eat_dir < 0: self.pt_eat_dir += 180
            if y < 0: self.pt_eat_dir += 180
            
            self.log('x = %d, y = %d, angle = %d'%(int(x), int(y), int(self.pt_eat_dir)))
            
            
        if self.interaction_mode == 'parametric_animation' and self.B1_was_pressed == True:
            self.paranim_start_time = time.time()  # this inits the animation
            
            # parameters based on click
            x = event.x - self.B1_pressed_at[0]
            y = -(event.y - self.B1_pressed_at[1])
            self.paranim_angle = np.arctan(y/(x+0.00001))*180/np.pi
            if self.paranim_angle < 0: self.paranim_angle += 180
            if y < 0: self.paranim_angle += 180
            
            self.paranim_x = self.B1_pressed_at[1]
            self.paranim_y = self.B1_pressed_at[0]
            
            # parameters based on selected animation
            f = open('parametric_animations/%s.anim'%self.ptact_var.get(), 'r')
            self.paranim_str = f.read()
            f.close()
            
            # parameters based on variables
            self.paranim_variables = self.get_paranim_vars()
            # self.paranim_variables = pixelburn2_rnd_variables()
            
            # parameters based on input
            self.paranim_length_ms = 1000*int(self.eatspeed_var.get())
            
            self.log('x = %d, y = %d, angle = %d'%(int(self.paranim_x), int(self.paranim_y), int(self.paranim_angle)))
            
        self.B1_was_pressed = False    
    
 
    def create_point(self, x, y):
        self.pt_point = (x, y)
        time_wait = float(self.eatwait_var.get())
        self.pt_eat_stage = -time_wait*float(self.eatspeed_var.get())/100
    
    
    def draw_oncamera(self, x, y):
        # print('x = %d, y = %d'%(event.x, event.y))
        # x = event.x
        # y = event.y - 76  # to account for bigger canvas than camera image
        
        # self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.camera_draw = cv2.circle(self.camera_draw, (x, y), self.brush_size, (255, 255, 255), -1)
              
    
    def debug_saveimg(self):
        i = int(np.random.random()*100)
        
        ret, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        np.save(self.camera_draw, 'temp/camera_draw_%d'%i)
        np.save(self.projector_arr, 'temp/projector_arr_%d'%i)
        np.save(self.cv2image, 'temp/camera_frame_%d'%i)
  
        
    def load_calib(self):
        self.homomatrix = get_homography_matrix()
        self.log('Calibration done!')


    def start_calib(self):
        global projector_calib_c
        projector_calib_c = -1
        self.frame_counter = 0


    def proj_calib_old(self):
        # here there are two counters:
        # projector_calib_c - counts where the dots are places
        # self.frame_counter - counts frames passed between putting the dot on projector screen and saving the camera image
        
        global projector_calib_c, inc_x, inc_y        
        
        # at the beginning, make the projector screen black
        if projector_calib_c == -1 and self.frame_counter == 0:
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        
        # if it is done with baseline (projector_calib_c > -1), at frame_counter == 0 start putting dots on projector array
        if projector_calib_c > -1 and self.frame_counter == 0:
            self.proj_x, self.proj_y = num_to_coords(projector_calib_c, size = calib_dots_dim)
        
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
            self.projector_arr = cv2.circle(self.projector_arr, (self.proj_x, self.proj_y), 18, (255, 255, 255), -1)
        
        # when N frames passed after last projector array modification (to give time for camera to update image) read and save the camera image 
        if self.frame_counter == 50:
            # ret, frame = cap.read()
            frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(448, 800, 3)
            # frame = self.cam_reader.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            if projector_calib_c > -1:  # meaining its done with the baseline                
                np.save('calibration/num%d'%projector_calib_c, frame)
                self.log('saved calibration/num%d'%projector_calib_c)
            else:
                np.save('calibration/baseline', frame)
                self.log('saved baseline')
        
            projector_calib_c += 1
            self.frame_counter = 0
        else:
            self.frame_counter += 1
          
          
    def proj_calib(self):
        # here there are two counters:
        # projector_calib_c - counts where the dots are places
        # self.frame_counter - counts frames passed between putting the dot on projector screen and saving the camera image
        
        global projector_calib_c, inc_x, inc_y        
        
        # at the beginning, make the projector screen black
        if projector_calib_c == -1 and self.frame_counter == 0:
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        
        # if it is done with baseline (projector_calib_c > -1), at frame_counter == 0 start putting dots on projector array
        if projector_calib_c > -1 and self.frame_counter == 0:
            self.proj_x, self.proj_y = num_to_coords(projector_calib_c, size = calib_dots_dim)
        
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
            self.projector_arr = cv2.circle(self.projector_arr, (self.proj_x, self.proj_y), 18, (255, 255, 255), -1)
        
        # when N frames passed after last projector array modification (to give time for camera to update image) read and save the camera image 
        if self.frame_counter == proj_calib_frames:
            # ret, frame = cap.read()
            frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(448, 800, 3)
            # frame = self.cam_reader.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            if projector_calib_c > -1:  # meaining its done with the baseline                
                np.save('calibration/num%d'%projector_calib_c, frame)
                self.log('saved calibration/num%d'%projector_calib_c)
            else:
                np.save('calibration/baseline', frame)
                self.log('saved baseline')
        
            projector_calib_c += 1
            self.frame_counter = 0
        else:
            self.frame_counter += 1
         
        
    def record_video(self):
        # start recording new video
        self.frame_lapse = 1
        if self.video_writer == None:
            # get the correct filename
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(data_dir + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.avi'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    out_path = data_dir + 'saved_images/' + fname + '_%s.avi'%str(num).zfill(2)
                    saved =1
                    
            # frame_y, frame_x = self.cam_reader.get_frame_shape()[:2]
            frame_y, frame_x = self.cv2image.shape[:2]
            # print("rec vid ", cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB).shape)
            self.video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_x, frame_y))
            # self.buttonRecord.configure(text = 'Stop recording')
            self.buttonRecord.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
            
        # if it is already recording, stop it
        else:
            self.video_writer.release()
            self.video_writer = None
            # self.buttonRecord.configure(text = 'Record video')
            self.buttonRecord.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
            
            
    def raw_enable(self):
        if self.raw_projector_screen == False:
            self.raw_projector_screen = True
        else:
            self.raw_projector_screen = False
            
        if self.raw_projector_screen == True:
            self.btn_raw_enable.configure(relief = SUNKEN)
        if self.raw_projector_screen == False:
            self.btn_raw_enable.configure(relief = RAISED)

    
    def record_timelapse(self, fname = ''):
        # start recording new video
        self.frame_lapse = 6
        if self.video_writer == None:
            # get the correct filename
            if fname == '':
                fname = self.path.get()
                if fname == '': fname = 'unnamed'                
                all_files = os.listdir(data_dir + '/saved_images/')
                saved = 0
                num = 0
                while saved == 0:
                    if fname + '_%s.avi'%str(num).zfill(2) in all_files:
                        num += 1
                    else:
                        out_path = data_dir + 'saved_images/' + fname + '_%s.avi'%str(num).zfill(2)
                        saved =1
            else:
                out_path = fname
                    
            # frame_y, frame_x = self.cam_reader.get_frame_shape()[:2]
            frame_y, frame_x = self.cv2image.shape[:2]
            # this is a little errory
            frame_y = 548
            frame_x = 800
            self.log('Beginning timelapse recording (%d, %d size)'%(frame_y, frame_x))
            self.video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_x, frame_y))
            # self.buttonRecord.configure(text = 'Stop recording')
            self.buttonRecord2.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
            
        # if it is already recording, stop it
        else:
            self.video_writer.release()
            self.video_writer = None
            # self.buttonRecord.configure(text = 'Record video')
            self.buttonRecord2.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
    
    
    def rec_data_button(self):
        if self.rec_data == False:
            self.rec_data = True
            self.buttonrecData.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
        else:
            self.rec_data = False
            self.buttonrecData.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
            
        
    def save_data_button(self):
        self.save_data = True
    
    
    def elli_refresh(self, value):
        # value = ell_com_var2.get()
        self.elliptec = connect_to_elliptec(int(value[3]))
    
        if self.elliptec == None:        
            self.label_ell_status2.config(text = 'not connected', bg='red')
        else:
            self.label_ell_status2.config(text = 'connected', bg='lime')
            
            
    def elli_refresh_top(self, value):
        # value = ell_com_var.get()
        self.elliptec_top = connect_to_elliptec_top(int(value[3]))
    
        if self.elliptec_top == None:        
            self.label_ell_status2_top.config(text = 'not connected', bg='red')
        else:
            self.label_ell_status2_top.config(text = 'connected', bg='lime')
    
    
    def elli6_refresh(self, value):
        self.ell6k = connect_to_ell6k(int(value[3]))
    
        if self.ell6k == None:        
            self.label_ell6_status2.config(text = 'not connected', bg='red')
        else:
            self.label_ell6_status2.config(text = 'connected', bg='lime')
    
    
    def btnAct_ell6_in(self):
        if self.ell6k != None:
            self.ell6k.write(b'0fw')
            self.ell6_state = 'IN'
            self.lab_ell6_state.config(text = 'ELL6 state = %s'%self.ell6_state)
        else:
            messagebox.showwarning(title='ELL6K', message='ELL6K is not connected! To insert mirror connect to ELL6K first.')
    
    
    def btnAct_ell6_out(self):
        if self.ell6k != None:
            self.ell6k.write(b'0bw')
            self.ell6_state = 'OUT'
            self.lab_ell6_state.config(text = 'ELL6 state = %s'%self.ell6_state)
        else:
            messagebox.showwarning(title='ELL6K', message='ELL6K is not connected! To insert mirror connect to ELL6K first.')
    
    
    def grbl_refresh(self, value):
        self.grbl = connect_to_grbl(int(value[3]))
    
        if self.grbl == None:        
            self.label_grbl_status2.config(text = 'not connected', bg='red')
        else:
            self.label_grbl_status2.config(text = 'connected', bg='lime')
    
    
    def thrm_refresh(self, value):
        self.thortemp = connect_to_thortemp(int(value[3]))
    
        if self.thortemp == None:        
            self.label_thrm_status2.config(text = 'not connected', bg='red')
        else:
            self.label_thrm_status2.config(text = 'connected', bg='lime')
    
            
    def btn_rigolcon(self):
        if self.rigol != None:
            messagebox.showinfo(title='Rigol', message='Rigol is already connected?')
        else:
            rm = pyvisa.ResourceManager()
            try:
                inst = rm.open_resource('USB0::0x1AB1::0x0643::DG8A220800267::INSTR')
                if inst.query("*IDN?")[:18] == 'Rigol Technologies':
                    self.rigol = inst
                    inst.write(':SOUR1:APPL:SQU 2000,5,2.5,0')
            except:
                messagebox.showerror(title='Rigol', message='Connection to Rigol failed!')
                self.rigol = None
    
        if self.rigol == None:        
            self.label_rigol_status2.config(text = 'not connected', bg='red')
        else:
            self.label_rigol_status2.config(text = 'connected', bg='lime')
            
    
    def set_temperature(self):
        thortemp_set_temp(self.thortemp, float(self.set_temp_var.get()))        
        self.esc_key_btn(1)
    
    
    def rotate_elli(self, ang_rel):
        '''
        rotate elliptec to selected angle (relative angle)
        and set all necessary display options
        '''        
        # set elliptec
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
        # ang_abs = ang_rel + self.ell_offset_bottom
        command = angle_to_ellocommand(ang_abs)
        self.elli_bottom_abs = ang_abs
        self.elliptec.write(command)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_ell_bottom.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_bottom))
        
        time.sleep(0.2)
        
        
    def rotate_elli_top(self, ang_rel):
        '''
        rotate elliptec to selected angle (relative angle)
        and set all necessary display options
        '''        
        # set elliptec
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_top)
        # ang_abs = ang_rel + self.ell_offset_bottom
        command = angle_to_ellocommand(ang_abs)
        self.elli_bottom_abs = ang_abs
        self.elliptec_top.write(command)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2_top.set(ang_rel)
        
        # write to text
        self.label_ell_top.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_top))
        
        time.sleep(0.2)
            
    
    def rotate_elli_rel(self):
        # get angles
        ang_rel = float(self.ell_var2.get())
        
        # call rotate function and defocuf from text field
        self.rotate_elli(ang_rel)
        self.buttonRotate2.focus_set()
        
        
    def rotate_elli_rel_top(self):
        # get angles
        ang_rel = float(self.ell_var2_top.get())
        
        # call rotate function and defocuf from text field
        self.rotate_elli_top(ang_rel)
        self.buttonRotate2.focus_set()
        
        
    def rotate_elli_step(self, value):
        if self.camera_image_type == 'MAP':
            self.ell_offset_bottom += value
        else:
            # get angle
            ang_rel = float(self.ell_var2.get())
            
            # change angle and write
            ang_rel = np.round(ang_rel+value, 5)
            ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
            # ang_abs = ang_rel + self.ell_offset_bottom
            command = angle_to_ellocommand(ang_abs)
            self.elli_bottom_abs = ang_abs
            self.elliptec.write(command)
            
            # recaluclate relative angle after change
            # ang_rel = angle_abs_to_rel(ang_abs, self.ell_offset_bottom)
            
            # write to textbox variables
            # self.ell_var.set(ang_abs)
            self.ell_var2.set(ang_rel)
            
            # write to text
            self.label_ell_bottom.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_bottom))


    def rotate_elli_step_top(self, value):
        # get angle
        ang_rel = float(self.ell_var2_top.get())
        
        # change angle and write
        ang_rel = np.round(ang_rel+value, 5)
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_top)
        # ang_abs = ang_rel + self.ell_offset_bottom
        command = angle_to_ellocommand(ang_abs)
        self.elli_top_abs = ang_abs
        self.elliptec_top.write(command)
        
        # recaluclate relative angle after change
        # ang_rel = angle_abs_to_rel(ang_abs, self.ell_offset_bottom)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2_top.set(ang_rel)
        
        # write to text
        self.label_ell_top.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_top))


    def elli_set_zero(self):  # obsolete?
        ang_rel = float(self.ell_var2.get())
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
        self.ell_offset_bottom = ang_abs
        self.log('Setting self.ell_offset_bottom to %f'%self.ell_offset_bottom)
        self.log('ang_rel = %f'%ang_rel)
        
        
    def elli_bottom_set_offset(self):
        self.ell_offset_bottom = self.elli_bottom_abs - 90
        self.log('Setting self.ell_offset_bottom to %f'%self.ell_offset_bottom)
        
        # write to text
        self.label_ell_bottom.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(self.ang_rel, self.ang_abs, self.ell_offset_bottom))
        
        
    def elli_top_set_offset(self):
        self.ell_offset_top = self.elli_top_abs
        self.log('Setting self.ell_offset_top to %f'%self.ell_offset_top)
        
        # write to text
        self.label_ell_top.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(self.ang_rel, self.ang_abs, self.ell_offset_top))

    
    def esc_key_btn(self, value):        
        self.buttonRotate2.focus_set()
    

    def image_move(self, event):
        self.log('x=%d, y=%d'%(event.x, event.y))
    
        
    def save_img(self):
        self.save_image = True

      
    def wait_timer(self, time_sec):
        '''
        sets waiting time in seconds, check if done using is_wait_done
        '''
        self.time_set_at = datetime.datetime.now()
        self.timer_wait_secs = time_sec
        
        
    def is_wait_done(self):
        if (datetime.datetime.now()-self.time_set_at).total_seconds() > self.timer_wait_secs:
            return True
        return False


    def name_the_file(self, add = '', date = None, polar = None):
        # acquire required data
        if polar == None:
            polar = float(self.ell_var2.get())
        self.ask_grbl_position()  # self.grblX, self.grblY
        if date == None:
            date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
        fname = self.path.get()
        
        if add == '':
            name = '%s %s P%2.1f X%2.3f Y%2.3f.npy'%(date, fname, polar, self.grblX, self.grblY)
        else:
            name = '%s %s %s P%2.1f X%2.3f Y%2.3f.npy'%(date, fname, add, polar, self.grblX, self.grblY)            
        return name
      

    def make_rchannel_map(self):
        params = self.advMapOptions.get()
        params = params.replace(' ', '')
        params += ','
        # text = 'G - gauss sigma, M - maxfilter sigma, Df - degradient resize factor
        # CF - compensation factor'
        sg = int(get_amap_parameter(params, 'G'))
        mx = int(get_amap_parameter(params, 'M'))
        dg = int(get_amap_parameter(params, 'Df'))
        cf = float(get_amap_parameter(params, 'Cf'))
                       
        self.advanced_mapka = get_channel_map(self.amap_frameM, self.amap_frameP, factor=cf, mx = mx, sg = sg)
        self.amap_ranger = 0.06


    def read_last_baseline(self, baseline_params):
        bdir = rmd + 'baselines/'
        baselines = [f for f in os.listdir(bdir) if f.find('baselineM ' + baseline_params) >= 0]
                
        now = datetime.datetime.now()
        if len(baselines) > 0:
            times = []
            for i in range(0, len(baselines)):
                date = datetime.datetime.strptime(baselines[i][:19], '%d-%m-%Y %H.%M.%S')
                times.append((now-date).total_seconds())
            
            arg = np.array(times).argmin()
            last_baselineM = np.load(bdir + baselines[arg])
            last_baselineP = np.load(bdir + baselines[arg].replace('baselineM', 'baselineP'))
            
            return last_baselineM, last_baselineP, times[arg]
        return -1, -1, -1
            

    def btn_get_baseline(self):
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [
                ['BASELINE'],
                ['END']
            ]        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
 

    def btn_bmap(self):
        self.log('Jamnik!\njbhgu783y5ugnui n84t')


    def make_position_list(self):
        # populate position list
        self.cpos = 0
        self.position_list = []
        stepsX = int(self.arealc_stepsX.get())
        stepsY = int(self.arealc_stepsY.get())
        off_x = float(self.arealc_x.get())
        off_y = float(self.arealc_y.get())
        
        for ix in range(0, stepsX):
            for iy in range(0, stepsY):
                self.position_list.append([ix*off_x, iy*off_y])

        
    def autop_areal_map(self):  
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [
            ['GOTO', [0, 0]],
            ['D_CURDATE']
        ]
        
        self.areal_map_polarlist = [int(p) for p in self.areal_pols_var.get().replace(' ', '').split(',')]
        for polar in self.areal_map_polarlist:
            key_name = 'gain_at_%s'%(str(polar).zfill(2))
            if key_name in self.extvars.keys():
                gain = int(self.extvars[key_name])
                self.cur_command_list.append(['D_CHANGE_GAIN', gain])
                
            # self.cur_command_list.append(['AREAL_GET_FRAMES', polar])
            self.cur_command_list.append(['MM_MAP', polar])
            # self.cur_command_list.append(['D_WAIT', 1])
            self.cur_command_list.append(['D_SAVEDATA', 'mapping'])
            
        
        self.cur_command_list.append(['NEXTPOS', 1])
        self.cur_command_list.append(['D_SET_CCOM_INDEX', 2])
        self.cur_command_list.append(['D_COMBINE_MAPS'])
        self.cur_command_list.append(['END'])
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
        
        
    def autop_find_offset(self):  
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = []

        for ang in range(0, 171, 10):
            self.cur_command_list.append(['D_SETPOL', ang + 85])
            self.cur_command_list.append(['D_SET_WAIT', 0.15])
            self.cur_command_list.append(['D_WAIT'])
            self.cur_command_list.append(['D_SETPOL_TOP', ang])
            self.cur_command_list.append(['D_SET_WAIT', 1])
            self.cur_command_list.append(['D_WAIT'])
            self.cur_command_list.append(['D_SAVEFRAME_OFFSETS', ang])
            
        self.cur_command_list.append(['END'])
        
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
        
        
    def autop_crosspolarize(self):
        self.crosspolarize_angle_list = range(0, 360, 10)
        self.crp_pos = 0
        # self.crp_file = file
        
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [
            ['POLARLIST_MEASURE'],
            ['D_POLARLIST_CHECK', -2],
            ['END']
        ]     
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
   

    def autop_areal_map_raw(self):  
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [
            ['GOTO', [0, 0]],
            ['D_CURDATE'],
            ['D_SAVEFRAME', 'mapping'],
            ['NEXTPOS', 1],
            ['D_CCOM_INDEX', -3],
            ['D_COMBINE_MAPS'],
            ['END']
        ]     
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)

    
    def btn_pattern_array(self):
        self.make_position_list()
        self.illuminate_pattern = False
        ill_t = int(self.set_ill_var.get())
        
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [          
            ['GOTO', [0, 0]],
            ['ILLUMINATE_PATTERN', ill_t],
            ['NEXTPOS', 1],
            ['D_CCOM_INDEX', -3],
            ['END']
        ]     
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
        
        
    def btn_pattern_array2(self):
        self.illuminate_pattern = False
        
        # turn off the laser if it is on
        if self.laserstate == 'ON': self.btn_laser_switch()
        
        # first load textfile
        f = open('illumination_programs/program1.txt', 'r')
        cnt = f.read()
        f.close()
        
        cnt = cnt.splitlines()
        
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = []
        
        for line in cnt:
            cline = line.split(',')
            pos_x = float(cline[0])
            pos_y = float(cline[1])
            time_s = float(cline[2])
            laser_power = float(cline[3])
            
            self.cur_command_list.append(['GOTO', [pos_x, pos_y]])
            self.cur_command_list.append(['D_ILLUMINATE', laser_power])
            self.cur_command_list.append(['D_SET_WAIT', time_s])
            self.cur_command_list.append(['D_WAIT'])
            self.cur_command_list.append(['D_ILLUMINATE', 0])
        
        self.cur_command_list.append(['END'])
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)


    def ill_program(self):
        self.illuminate_pattern = False
        
        # turn off the laser if it is on
        if self.laserstate == 'ON': self.btn_laser_switch()
        
        # switch to image mode
        if self.interaction_mode != 'image': self.btn_image()
        
        # first load textfile
        fpath_excel = filedialog.askopenfilename()
        program_df = pd.read_excel(fpath_excel, engine='openpyxl', 
                          names = ['x', 'y', 'speed', 'laser', 'time', 'file', 'raw'])

        
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = []
        
        for i in range(0, len(program_df)):
            pos_x = float(program_df.iloc[i]['x'])
            pos_y = float(program_df.iloc[i]['y'])
            speed = float(program_df.iloc[i]['speed'])
            laser_p = float(program_df.iloc[i]['laser'])
            time_s = float(program_df.iloc[i]['time'])
            fpath = str(program_df.iloc[i]['file'])
            raw_out = str(program_df.iloc[i]['raw'])
            
            self.cur_command_list.append(['SET_GRBL_SPEED', speed])
            self.cur_command_list.append(['GOTO', [pos_x, pos_y]])
            if laser_p != 0:
                self.cur_command_list.append(['LOAD_IMG', fpath])
                if raw_out == 1: 
                    self.cur_command_list.append(['RAW_OUT', 1, time_s])
                    self.cur_command_list.append(['D_ILLUMINATE', laser_p])
                    self.cur_command_list.append(['D_RAW_WAIT_END'])
                else:
                    self.cur_command_list.append(['D_ILLUMINATE', laser_p])
                    self.cur_command_list.append(['D_SET_WAIT', time_s])
                    self.cur_command_list.append(['D_WAIT'])
                    self.cur_command_list.append(['D_ILLUMINATE', 0])
        
        self.cur_command_list.append(['END'])
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)


    def ill_program_commands(self):
        self.illuminate_pattern = False
        
        # turn off the laser if it is on
        if self.laserstate == 'ON': self.btn_laser_switch()
        
        # switch to image mode
        if self.interaction_mode != 'image': self.btn_image()
        
        # first load textfile
        fpath_excel = filedialog.askopenfilename()
        program_df = pd.read_excel(fpath_excel, engine='openpyxl', 
                          names = ['x', 'y', 'speed', 'laser', 'time', 'file', 'raw'])

        
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = []
        
        for i in range(0, len(program_df)):
            pos_x = float(program_df.iloc[i]['x'])
            pos_y = float(program_df.iloc[i]['y'])
            speed = float(program_df.iloc[i]['speed'])
            laser_p = float(program_df.iloc[i]['laser'])
            time_s = float(program_df.iloc[i]['time'])
            fpath = str(program_df.iloc[i]['file'])
            raw_out = str(program_df.iloc[i]['raw'])
            
            self.cur_command_list.append(['SET_GRBL_SPEED', speed])
            self.cur_command_list.append(['GOTO', [pos_x, pos_y]])
            if laser_p != 0:
                self.cur_command_list.append(['LOAD_IMG', fpath])
                if raw_out == 1: 
                    self.cur_command_list.append(['RAW_OUT', 1, time_s])
                    self.cur_command_list.append(['D_ILLUMINATE', laser_p])
                    self.cur_command_list.append(['D_RAW_WAIT_END'])
                else:
                    self.cur_command_list.append(['D_ILLUMINATE', laser_p])
                    self.cur_command_list.append(['D_SET_WAIT', time_s])
                    self.cur_command_list.append(['D_WAIT'])
                    self.cur_command_list.append(['D_ILLUMINATE', 0])
        
        self.cur_command_list.append(['END'])
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)


    def btn_flip(self):
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        self.cur_command_list = [
            ['GOTO', [0, 0]],
            ['D_CURDATE'],
            ['MM_MAP'],
            ['D_SAVEDATA', 'before'],
            ['D_CALC_SHOT_ARR', int(self.eatdir_var.get())],
            ['FLIP'],
            ['MM_MAP'],
            ['D_SAVEDATA', 'after'],
            ['D_EVALUATE_MAP', 0, -8],  # go to beginning
            ['NEXTPOS', 1],
            ['D_CCOM_INDEX', -10],
            ['END']
        ]   
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)


    def autotune_anim_params(self):
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        
        self.cur_command_list = []
        for pos in self.position_list:
            self.cur_command_list.append(['GOTO', [pos[0], pos[1]]])
            self.cur_command_list.append(['D_CURDATE'])  # remembers the date at which this command will be run
            self.cur_command_list.append(['MM_MAP'])  # makes map (remembers it as self.advanced_mapka)
            self.cur_command_list.append(['D_SAVEDATA', 'before'])  # save frames
            self.cur_command_list.append(['D_SAVEMAP', 'before'])  # save mapka
            self.cur_command_list.append(['D_RANDOMIZE_PARANIM'])  # randomizes parameters
            self.cur_command_list.append(['MAKE_PARANIM'])  #  run paranim, save the parameters
            self.cur_command_list.append(['MM_MAP'])  # makes map (remembers it as self.advanced_mapka)
            self.cur_command_list.append(['D_SAVEDATA', 'after'])  # save frames
            self.cur_command_list.append(['D_SAVEMAP', 'after'])  # save mapka
        
        self.cur_command_list.append(['END'])  
            
            

        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
        
        
    def btn_autofocus(self):
        self.camera_image_type = 'AUTOPILOT'
        
        self.afocus_pos = []
        self.afocus_fcs = []
        self.afocus_step = -0.2
        self.afocus_round = 0
        self.afocus_zpos = 0
        
        self.cur_command_list = []
        self.cur_command_list.append(['AUTOFOCUS2'])
        self.cur_command_list.append(['END'])  
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
        
        
    def btn_autofocus_fine(self):
        self.camera_image_type = 'AUTOPILOT'
        
        self.afocus_pos = []
        self.afocus_fcs = []
        self.afocus_step = -0.02
        self.afocus_round = 0
        self.afocus_zpos = 0
        
        self.cur_command_list = []
        self.cur_command_list.append(['AUTOFOCUS2'])
        self.cur_command_list.append(['END'])  
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)


    def autotune_single(self):
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        
        self.cur_command_list = []
        
        self.cur_command_list.append(['D_CURDATE'])  # remembers the date at which this command will be run
        self.cur_command_list.append(['MM_MAP'])  # makes map (remembers it as self.advanced_mapka)
        self.cur_command_list.append(['D_SAVEDATA', 'before'])  # save frames
        self.cur_command_list.append(['D_SAVEMAP', 'before'])  # save mapka
        self.cur_command_list.append(['MAKE_PARANIM'])  #  run paranim, save the parameters
        self.cur_command_list.append(['MM_MAP'])  # makes map (remembers it as self.advanced_mapka)
        self.cur_command_list.append(['D_SAVEDATA', 'after'])  # save frames
        self.cur_command_list.append(['D_SAVEMAP', 'after'])  # save mapka
        
        self.cur_command_list.append(['END'])  
            
            

        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
 
 
    def start_patterning(self):
        self.make_position_list()
    
        self.camera_image_type = 'AUTOPILOT'
        
        self.pixel_current = int(self.extvars['pixel_start'])
        color, row, col = get_pattern_pixel(self.current_pattern, self.pixel_current) 
        self.pixel_x = float(self.arealc_x.get())*col
        self.pixel_y = float(self.arealc_y.get())*row  
        self.filename_numa = 0
        self.pixelburns_in_row = 0
        self.burn_hist_gfs = []
        self.burn_hist_anims = []
        # img = Image.open('patterns/smile.png')
        # arr = np.array(img)
        # self.current_pattern = arr[:,:,0]/255
        
        self.cur_command_list = []
        
        self.cur_command_list.append(['GOTO', 'pixel'])     
        self.cur_command_list.append(['D_CURDATE'])  # remembers the date at which this command will be run
        self.cur_command_list.append(['MM_MAP', self.polrot])  # makes map (remembers it as self.advanced_mapka)
        self.cur_command_list.append(['D_SAVEDATA', 'numa'])  # save frames
        self.cur_command_list.append(['D_SAVEMAP', 'numa'])  # save mapka
        self.cur_command_list.append(['D_CHECK_PIXEL'])  #  check if burn here (if yes advance self.pixel_current [+1], reset numa, if no repeat all)
        self.cur_command_list.append(['MAKE_PARANIM'])
        self.cur_command_list.append(['D_SET_WAIT', 3])
        self.cur_command_list.append(['D_WAIT'])
        self.cur_command_list.append(['REPEAT'])   


        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
      
    
    def btn_mmap(self):
        # resets the mmap variables (they are floats!)
        self.mmap_plus = np.zeros((448, 800, 3), float)
        self.mmap_minus = np.zeros((448, 800, 3), float)
        self.mm_count = int(self.mm_var.get())
    
        self.camera_image_type = 'AUTOPILOT'
        
        self.cur_command_list = []
        
        # self.cur_command_list.append(['D_SETPOL', 90 + self.polrot])
        # self.cur_command_list.append(['D_SET_WAIT', 1.0])
        # self.cur_command_list.append(['D_WAIT'])
        # for i in range(0, self.mm_count):
            # self.cur_command_list.append(['D_MMAP_ADD_FRAME_P'])

        # self.cur_command_list.append(['D_SETPOL', 90 - self.polrot])
        # self.cur_command_list.append(['D_SET_WAIT', 1.0])
        # self.cur_command_list.append(['D_WAIT'])
        # for i in range(0, self.mm_count):
            # self.cur_command_list.append(['D_MMAP_ADD_FRAME_M'])
            
        # self.cur_command_list.append(['D_MMAP_MAKE'])
        self.cur_command_list.append(['MM_MAP', self.polrot])
        self.cur_command_list.append(['HOLD'])
        
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_map_counter = 0
        self.ctask_index = 0
        self.ccom_index = 0
        self.task_list = [['GETNEW']]  # GETNEW
        self.esc_key_btn(1)
             
        # reset the map
        self.advanced_mapka = np.zeros(self.advanced_mapka.shape, float)
    
    
    def show_frame(self):    
        # frame, frame_is_new = self.cam_reader.get_frame()
        # if __name__ == '__main__':
            # frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(448, 800, 3)
        # else:
            # frame = np.zeros((800, 448, 3), np.uint8)
        frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(448, 800, 3)
        self.bright_cnt += frame.sum()
        
        # if self.camera_should_change_gain == True:
            # self.cam_reader.change_gain(self.camera_gain)
            # self.camera_should_change_gain = False
            
        # if self.camera_should_change_expo == True:
            # self.cam_reader.change_expo(self.camera_expo)
            # self.camera_should_change_expo = False
            
        # new camera gain/expo change
        # self.config_arr[0] = self.camera_gain
        # self.config_arr[1] = self.camera_expo
            
            
        # every tenth frame action
        if self.frame10counter >= 10:
            self.brightness = self.bright_cnt/10
            self.bright_cnt = 0
            # thermal read actual
            if self.thortemp != None:
                self.thortemp_Tactual = thortemp_get_temp(self.thortemp)
                self.lab_tempcur.config(text = 'T = %3.1f °C'%self.thortemp_Tactual)
                
                
            # XBOX controller pad actions            
            if self.xbox_controller != None:  
                control_data = self.xbox_controller.read()
                
                # moving grbl
                if self.grbl != None:
                    gx, gy = int(10*control_data[0])/10, -int(10*control_data[1])/10
                    if (gx != 0) or (gy != 0):
                        dist = np.sqrt(gx**2+gy**2)
                        s = dist/1.0*60
                        s = min(50, s)
                        print('going %2.4f at %2.4f mm/min'%(dist, s))
                        self.grbl_command('G21 G91 G1 X%1.2f Y%1.2f F%s\n'%(gx, gy, s))
                        
                # laser constant power shot
                if self.rigol != None:
                    if control_data[3] == 0:
                        if control_data[2] == 1 and self.laserstate == 'OFF':
                            self.btn_laser_switch()
                        if control_data[2] == 0 and self.laserstate == 'ON':
                            self.btn_laser_switch()
                        
                # laser variable power shot
                if self.rigol != None:
                    if control_data[2] == 0:
                        max_pow = float(self.laserduty_var_pad.get())
                        laser_pow = float('%2.1f'%(max_pow*control_data[3]))
                        if control_data[3] > 0 and self.laserstate == 'OFF':
                            self.laserduty_var.set(laser_pow)
                            self.btn_laserduty()
                            self.btn_laser_switch()
                        if control_data[3] == 0 and self.laserstate == 'ON':
                            self.btn_laser_switch()
                        if control_data[3] > 0 and self.laserstate == 'ON':
                            self.laserduty_var.set(laser_pow)
                            self.btn_laserduty()
            
                
            self.frame10counter = 0
        else:
            self.frame10counter += 1
    
        # frame_is_new = True
        # frame = np.zeros((448, 800, 3), np.uint8)
        
            
        # depending on camera_image_type, cv2image will be selected
        # no transformations
        if self.camera_image_type == 'RAW':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #A
            self.atp_display_map = False
        
        # dynamic just shows two opposite polarized images
        if self.camera_image_type == 'DYN':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # also no direct image transfromarmations 
            self.camera_map_counter += 1
            if self.camera_map_counter == self.frame_map_switch: 
                rel_ang = int(self.rotaround_var.get()) - self.polrot 
                self.rotate_elli(rel_ang)            
            if self.camera_map_counter == 2*self.frame_map_switch: 
                self.camera_map_counter = 0
                rel_ang = int(self.rotaround_var.get()) + self.polrot        
                self.rotate_elli(rel_ang)  
            
        # map shows difference between two polarizations (at the end of each cycle (from 0 to FMS and from FMS to 2*FMS) it calculates the map and switches the polarizer)
        if self.camera_image_type == 'MAP':
            self.camera_map_counter += 1
            params = self.advMapOptions.get()
            params = params.replace(' ', '')
            params += ','
            sg = int(get_amap_parameter(params, 'G'))
            ch = int(get_amap_parameter(params, 'CH'))
            
            if self.camera_map_counter == self.frame_map_switch-5:  # rotate elliptec few frames before image aquisition
                rel_ang = int(self.rotaround_var.get()) - self.polrot
                self.rotate_elli(rel_ang)
            if self.camera_map_counter == self.frame_map_switch: 
                # rel_ang = int(self.rotaround_var.get()) - self.polrot
                # self.rotate_elli(rel_ang)

                # norm_cur = gaussian_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)/255, sigma = self.map_gauss)
                norm_cur = frame.copy()
                norm_prev = self.prev_frame

                # self.cv2image = cmap_array(norm_cur - norm_prev - self.calib_factor*self.calib_grad, self.diff_cmap, rng = self.map_rng)
                self.cv2image = cmap_array(baseline_mapping(norm_cur, norm_prev, norm_prev*0, norm_prev*0, sigma = sg, ch = ch), self.diff_cmap, rng = self.map_rng)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
                
                self.cv2image_const = self.cv2image.copy()
                self.prev_frame = frame.copy()
                
            if self.camera_map_counter == 2*self.frame_map_switch-5: 
                # self.camera_map_counter = 0
                rel_ang = int(self.rotaround_var.get()) + self.polrot        
                self.rotate_elli(rel_ang)
            if self.camera_map_counter == 2*self.frame_map_switch: 
                self.camera_map_counter = 0
                # rel_ang = int(self.rotaround_var.get()) + self.polrot        
                # self.rotate_elli(rel_ang)
                
                norm_cur = frame.copy()
                norm_prev = self.prev_frame
                
                self.cv2image = cmap_array(baseline_mapping(norm_prev, norm_cur, norm_prev*0, norm_prev*0, sigma = sg, ch = ch), self.diff_cmap, rng = self.map_rng)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
                self.cv2image_const = self.cv2image.copy()
                # self.prev_grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame = frame.copy()
                
            if (self.camera_map_counter != 2*self.frame_map_switch) and (self.camera_map_counter != self.frame_map_switch):
                self.cv2image = self.cv2image_const.copy()
        
        if self.camera_image_type == 'AMAP' or self.camera_image_type == 'AMAP2':
            self.cv2image = frame.copy()
            # sequence of operations is as follows:
            # first rotate polarizer to plus position
            if self.camera_map_counter == 0:
                rel_ang = 90 + self.polrot        
                self.rotate_elli(rel_ang)
            
            # wait few frames and save current frame to memory then rotate the polarizer to minus position
            if self.camera_map_counter == 15:
                self.amap_frameP = frame.copy()
                rel_ang = 90 - self.polrot        
                self.rotate_elli(rel_ang)
                
            # save the minus frame
            if self.camera_map_counter == 30:
                self.amap_frameM = frame.copy()
            
            # calculate the map based on parameters read from text entry
            if self.camera_map_counter == 31:
                params = self.advMapOptions.get()
                params = params.replace(' ', '')
                params += ','
                # text = 'G - gauss sigma, M - maxfilter sigma, Df - degradient resize factor
                # CF - compensation factor'
                sg = int(get_amap_parameter(params, 'G'))
                mx = int(get_amap_parameter(params, 'M'))
                dg = int(get_amap_parameter(params, 'Df'))
                cf = float(get_amap_parameter(params, 'Cf'))
                
                if self.camera_image_type == 'AMAP':
                    self.advanced_mapka, self.amap_ranger = advanced_map(self.amap_frameM, self.amap_frameP, sg = sg, degrad_size = dg, maxi = mx)
                if self.camera_image_type == 'AMAP2':                        
                    self.advanced_mapka = get_channel_map(self.amap_frameM, self.amap_frameP, factor=cf, mx = mx, sg = sg)
                    self.amap_ranger = 0.06
            
            
            if self.camera_map_counter < 32:
                self.camera_map_counter += 1
            else:
                self.cv2image = cmap_array(self.advanced_mapka, self.diff_cmap, rng = self.amap_ranger)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
                
        if self.camera_image_type == 'SPOL':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # also no direct image transfromarmations
            
            if self.camera_map_counter == 0:
                self.spol_angle = 70
                self.rotate_elli(self.spol_angle)
                self.spol_date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                
            if self.camera_map_counter%50 == 49:
                # save current frame with current spol_angle
                fname = self.path.get()                
                fname = self.spol_date + ' ' + fname + ' (P%d)'%self.spol_angle
                path_name = data_dir + 'raw_mikro_data/' + fname
                np.save(path_name, frame)
                
                # rotate polarizer to new position
                self.spol_angle += 5
                self.rotate_elli(self.spol_angle)
            
            self.camera_map_counter += 1
            
            # when all interesting polarizations are reached, go back to raw             
            if self.spol_angle > 110:
                self.camera_image_type = 'RAW'
                
        if self.camera_image_type == 'AREAL':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # also no direct image transfromarmations
            # print(self.camera_map_counter)
            
            # at the beginning
            if self.camera_map_counter == 0:
                self.spol_date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                
                # calculate necessary positions [in um]
                self.pos_list = []  # obsolete system
                step_x = 0.5
                step_y = 0.5
                for ix in range(0, 3):
                    for iy in range(0, 3):
                        x = ix * step_x
                        y = iy * step_y
                        self.pos_list.append([x, y])
                
                # set the fisrt position, go to it
                self.current_pos_index = 0
                self.apx, self.apy = self.pos_list[self.current_pos_index]
                
                # go to start position
                self.ask_grbl_position()
                self.autop_finish_move_seconds = move_time([self.grblX, self.grblY], [self.apx, self.apy], float(self.grbl_speed.get())) + 1
                self.grbl_command('G21 G90 G1 X%2.2f Y%2.2f F%s\n'%(self.apx, self.apy, self.grbl_speed.get()))
                               
                # create folder for data (for debugging)
                self.ap_savedir = data_dir + 'raw_mikro_data/' + 'area ' + self.spol_date + '/'
                os.mkdir(self.ap_savedir)
                
                # set the current time
                self.autop_clock = datetime.datetime.now()
                self.camera_map_counter = 1
                
        
            if self.camera_map_counter == 1:
                # first check if enough time elapsed to finish last move command
                if (datetime.datetime.now()-self.autop_clock).total_seconds() > self.autop_finish_move_seconds:
                    # first save current frame
                    fname = self.path.get()                
                    fname = fname + ' (x=%2.2f y=%2.2f)'%(self.apx, self.apy)
                    np.save(self.ap_savedir + fname, frame)
                    self.cv2image = np.zeros((448, 800, 3), np.uint8)
                    
                    # advance to next position (if it is not last position)
                    if self.current_pos_index < len(self.pos_list)-1:
                        self.current_pos_index += 1
                        
                        # ask grbl to move to position
                        self.apx, self.apy = self.pos_list[self.current_pos_index]
                        
                        # send command to grbl
                        self.ask_grbl_position()
                        self.autop_finish_move_seconds = move_time([self.grblX, self.grblY], [self.apx, self.apy], float(self.grbl_speed.get())) + 1
                        self.autop_clock = datetime.datetime.now()
                        self.grbl_command('G21 G90 G1 X%2.2f Y%2.2f F%s\n'%(self.apx, self.apy, self.grbl_speed.get()))
                    else:
                        self.camera_image_type = 'RAW'
            
        if self.camera_image_type == 'AUTOPILOT':
            # here, display image may differ
            if self.atp_display_map == True: 
                self.cv2image = (cmap_array(self.advanced_mapka, self.diff_cmap, rng = self.amap_ranger)*255).astype(np.uint8)
                self.cv2image = cv2.resize(self.cv2image, (800, 448))
                # self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # no direct image transfromarmations
            else:
                self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # no direct image transfromarmations
            # command list will be set somewhere else

            command_list = self.cur_command_list
            # command_list = [
                # ['MAKEMAP', self.polrot, data_dir + 'raw_mikro_data/autop/', 'before'],
                # ['CALC_SHOT_ARR', 1, data_dir + 'raw_mikro_data/autop/'],
                # ['FLIP'],
                # ['MAKEMAP', self.polrot, data_dir + 'raw_mikro_data/autop/', 'after'],
                # ['END']
            # ]
            
            # prepare task list based on current command (ccom_index)
            ccom = command_list[self.ccom_index]
            
            # at the beginning of the loop, check which task is current
            self.ctask = self.task_list[self.ctask_index]
            
            if self.ctask[0] == 'GETNEW':  # as in get new task list (GETNEW is added at the end of each task_list after adding tasks from command)
                self.task_list = []
                self.log('autopilot: read command no. %d: %s'%(self.ccom_index, str(ccom)))
                
                # command translation
                if ccom[0] == 'GOTO':
                    # first calculate necessary time to reach designated position
                    self.ask_grbl_position()
                    if type(ccom[1]) != str:
                        t = move_time([self.grblX, self.grblY], ccom[1], float(self.grbl_speed.get())) + 1
                        self.task_list.append(['GOTO', ccom[1]])
                    else:  # pixel
                        t = move_time([self.grblX, self.grblY], (self.pixel_x, self.pixel_y), float(self.grbl_speed.get())) + 1
                        self.task_list.append(['GOTO', (self.pixel_x, self.pixel_y)])                        
                    self.task_list.append(['D_SET_WAIT', t])
                    self.task_list.append(['D_WAIT'])
                    
                if ccom[0] == 'GOTO_I':
                    # first calculate necessary time to reach designated position
                    self.ask_grbl_position()
                    newX = self.grblX + ccom[1][0]
                    newY = self.grblY + ccom[1][1]
                    t = move_time([self.grblX, self.grblY], [newX, newY], float(self.grbl_speed.get())) + 1
                    self.task_list.append(['GOTO', [newX, newY]])
                    self.task_list.append(['D_SET_WAIT', t])
                    self.task_list.append(['D_WAIT'])                    
                    
                if ccom[0] == 'SAVEPOLS':                    
                    for i in range(0, len(ccom[1])):
                        self.task_list.append(['D_SETPOL', ccom[1][i]])
                        self.task_list.append(['D_SET_WAIT', 1.5])
                        self.task_list.append(['D_WAIT'])
                        self.task_list.append(['SAVEFRAME', ccom[2]])
                   
                if ccom[0] == 'FLIP':
                    self.task_list.append(['D_IN_MIRROR'])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    
                    self.task_list.append(['D_LASER_STATE', 1])
                    self.task_list.append(['FLIP'])
                    self.task_list.append(['WAIT_FLIP'])
                    self.task_list.append(['D_LASER_STATE', 0])
                    
                    self.task_list.append(['D_OUT_MIRROR'])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    
                if ccom[0] == 'BASELINE':                    
                    self.task_list.append(['D_SETPOL', 90 - self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'M'])
                    self.task_list.append(['D_SETPOL', 90 + self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'P'])
                    self.task_list.append(['CALC_BASELINE'])
                    
                if ccom[0] == 'AUTOFOCUS':                    
                    self.task_list.append(['D_FOCUS_STEP'])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['D_FOCUS_AUTO_DECIDE'])                    
                    
                if ccom[0] == 'AUTOFOCUS2':
                    step_per_round = 15
                    if self.fcs == None:  # means it has to still search
                        self.afocus_step *= -1  # reverse direction
                        
                        # go back to position 0
                        # distance = step_per_round * self.afocus_step * self.afocus_round
                        distance = -self.afocus_zpos
                        self.task_list.append(['D_FOCUS_STEP', distance])
                        self.task_list.append(['D_SET_WAIT', focus_time(distance, 15)+0.5])
                        self.task_list.append(['D_WAIT'])
                        
                        for i in range(1, step_per_round+1):
                            self.task_list.append(['D_FOCUS_STEP', self.afocus_step])
                            self.task_list.append(['D_SET_WAIT', focus_time(self.afocus_step, 15)+0.1])
                            self.task_list.append(['D_WAIT'])
                            # pos = sign(self.afocus_step)*2*int(self.afocus_round/2) + i*self.afocus_step  # relative postion of Z
                            self.afocus_zpos += self.afocus_step
                            self.task_list.append(['D_AFOCUS_APPEND', self.afocus_zpos])
                            
                        self.task_list.append(['D_FOCUS_AUTO_DECIDE'])
                    else:  # focus was found, go to it
                        self.afocus_step *= -1  # reverse direction
                        distance = -self.afocus_zpos
                        
                        self.log('distance = %f'%distance)
                        self.log('fcs = %f'%self.fcs)
                        
                        self.task_list.append(['D_FOCUS_STEP', distance+self.fcs])
                        self.task_list.append(['D_SET_WAIT', focus_time(distance + self.fcs, 15)+0.5])
                        self.task_list.append(['D_WAIT'])
                        self.fcs = None
                        self.afocus_zpos = 0
                                            
                if ccom[0] == 'BASELINE_MAP':
                    self.task_list.append(['D_SETPOL', int(self.rotaround_var.get()) - self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'M'])
                    self.task_list.append(['D_SETPOL', int(self.rotaround_var.get()) + self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'P'])
                    self.task_list.append(['CALC_BMAP'])
                    
                if ccom[0] == 'AREAL_GET_FRAMES':
                    self.polrot = ccom[1]
                    self.task_list.append(['D_SETPOL', int(self.rotaround_var.get()) - self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'M'])
                    self.task_list.append(['D_SETPOL', int(self.rotaround_var.get()) + self.polrot])
                    self.task_list.append(['D_SET_WAIT', 1.0])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['REMFRAME', 'P'])
                                            
                if ccom[0] == 'REPEAT':
                    self.ccom_index = 0
                                        
                if ccom[0] == 'MM_MAP':
                    self.mmap_plus = np.zeros((448, 800, 3), float)
                    self.mmap_minus = np.zeros((448, 800, 3), float)
                    self.polrot = ccom[1]
                    
                    angles = [int(a) for a in self.map_rots_var.get().replace(' ', '').split(',')]
                    for ang in angles:
                        off = self.offsets[int(round_to(ang, 10)/10)]
                        # set top elliptec
                        self.task_list.append(['D_SETPOL_TOP', ang])
                        self.task_list.append(['D_SET_WAIT', 0.1])
                        self.task_list.append(['D_WAIT'])
                        
                        # set plus on bottom
                        self.task_list.append(['D_SETPOL', ang + 90 + self.polrot])
                        self.task_list.append(['D_SET_WAIT', 1.25])
                        self.task_list.append(['D_WAIT'])
                        for i in range(0, self.mm_count):
                            self.task_list.append(['D_MMAP_ADD_FRAME_P', [off[0], off[1]]])
                            
                        # set minus on bottom
                        self.task_list.append(['D_SETPOL', ang + 90 - self.polrot])
                        self.task_list.append(['D_SET_WAIT', 1.25])
                        self.task_list.append(['D_WAIT'])
                        for i in range(0, self.mm_count):
                            self.task_list.append(['D_MMAP_ADD_FRAME_M', [off[0], off[1]]])
                        
                    self.task_list.append(['D_MMAP_MAKE'])
                    
                if ccom[0] == 'ILLUMINATE_PATTERN':  # obsolete?
                    ill_t = int(self.set_ill_var.get())
                    self.task_list.append(['D_ILLUMINATE', 1])
                    self.task_list.append(['D_SET_WAIT', ill_t])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['D_ILLUMINATE', 0])
                    
                if ccom[0] == 'MAKE_PARANIM':
                    # get variables from places
                    self.paranim_variables = self.get_paranim_vars()
                    self.paranim_length_ms = 1000*int(float(self.eatspeed_var.get()))
                    temp = float(self.set_temp_var.get())
                    
                    # check map where to leave a spot
                    if self.ptact_var.get()[:-1] == 'pixel-burn':
                        color, row, col = get_pattern_pixel(self.current_pattern, self.pixel_current) 
                        if color == 1: vars_xy = select_side(self.advanced_mapka, 1, check_size = int(self.extvars['pixel_check_size']), side_thresh = float(self.extvars['side_threshold']), thresh = float(self.extvars['check_threshold']))
                        if color == 0: vars_xy = select_side(self.advanced_mapka, -1, check_size = int(self.extvars['pixel_check_size']), side_thresh = float(self.extvars['side_threshold']), thresh = float(self.extvars['check_threshold']))
                        self.paranim_variables.append(vars_xy[0])
                        self.paranim_variables.append(vars_xy[1])
                    
                    # save variables to file
                    numa = str(self.filename_numa - 1).zfill(3)
                    fname = rmd + self.name_the_file(add = 'burnParams_'+numa, date=self.savedata_date, polar = 90 - self.polrot)
                    f = open(fname, 'w')
                    f.write('anim_type=%s\n'%(self.ptact_var.get()))
                    for i, v in enumerate(self.paranim_variables):
                        f.write('v%d=%2.4f\n'%(i, v))
                    f.write('length=%2.4f\n'%(self.paranim_length_ms/1000))
                    f.write('power=%2.4f\n'%self.laserduty)
                    f.write('size=%d\n'%self.brush_size)
                    f.write('temp=%d\n'%temp)
                    f.write('paranim_x=%d\n'%self.paranim_x)
                    f.write('paranim_y=%d\n'%self.paranim_y)
                    f.write('paranim_a=%d\n'%self.paranim_angle)
                    f.write('paranim_pixel_check_size=%d\n'%int(self.extvars['pixel_check_size']))
                    f.write('check_threshold=%2.4f\n'%float(self.extvars['check_threshold']))
                    f.write('direction=%d\n'%self.kolo_direction)
                    f.write(self.burn_params_comment)
                    f.close()
                    
                    # parameters based on selected animation
                    f = open('parametric_animations/%s.anim'%self.ptact_var.get(), 'r')
                    self.paranim_str = f.read()
                    f.close()
                    
                    self.task_list.append(['D_SET_TEMP', temp])  # sets the temperature
                    self.task_list.append(['D_WAIT_TEMP', 1.5])  # waits for the temperature
                    self.task_list.append(['D_SET_WAIT', 3])  # wait few seconds more to stabilise the temp
                    self.task_list.append(['D_WAIT'])
                    
                    self.task_list.append(['D_IN_MIRROR'])
                    self.task_list.append(['D_SET_WAIT', 1])  # wait for mirror
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['D_CHANGE_GAIN', self.camera_lo_gain])
                    
                    self.task_list.append(['D_START_PARANIM'])
                    self.task_list.append(['D_ILLUMINATE', self.laserduty])  # turns on the laser with selected power
                    # self.task_list.append(['D_SET_WAIT', self.paranim_length_ms/1000+1])
                    self.task_list.append(['D_WAIT_PARANIM'])
                    self.task_list.append(['D_ILLUMINATE', 0])
                    
                    self.task_list.append(['D_OUT_MIRROR'])
                    self.task_list.append(['D_CHANGE_GAIN', self.camera_hi_gain])
                    
                if ccom[0] == 'POLARLIST_MEASURE':
                    self.task_list.append(['D_POLARLIST'])  # directly rotate elliptec by angle from list as absolute position
                    self.task_list.append(['D_SET_WAIT', 0.5])
                    self.task_list.append(['D_WAIT'])
                    self.task_list.append(['D_MEASURE_CRP_INT'])  # measures total intensity of light and saves to file
                    
                if ccom[0] == 'LOAD_IMG':
                    self.load_image_path(ccom[1])
                    
                if ccom[0] == 'RAW_OUT':
                    if ccom[1] == 1:
                        self.rawdelay_var.set(ccom[2])
                        if self.raw_projector_screen == False:
                            self.raw_enable()
                    
                if ccom[0] == 'SET_GRBL_SPEED':
                    self.grbl_speed.set(str(ccom[1]))
                    
                if ccom[0] == 'NEXTPOS':
                    if self.cpos+1 == len(self.position_list):
                        self.task_list.append(['D_CCOM_INDEX', ccom[1]])
                    else:
                        self.task_list.append(['NEXTPOS'])
                        self.task_list.append(['D_WAIT'])  # here, wait time is set by NEXTPOS (no need to use D_SET_WAIT)
                      
                if ccom[0] == 'END':
                    self.camera_image_type = 'RAW'
                    self.atp_display_map = False
                    self.log('autopilot: END command, going to RAW mode!')
                    
                    
                    
                    
                # direct commands
                if ccom[0][:2] == 'D_':
                    self.task_list.append(ccom)
                    
                if ccom[0] != 'HOLD':
                    self.task_list.append(['GETNEW'])
                    self.ctask_index = 0
                    self.ccom_index += 1
                    for ttt in self.task_list:
                        if len(ttt) > 0: self.log('autopilot new task list:%s'%str(ttt))
                else:
                    self.ctask_index = 0
                    self.task_list = [[None]]
            
            # task translation
            if self.ctask[0] == 'D_CHECK_PIXEL':
                self.ctask_index += 1
                self.log('------- autopilot task report: checking pixel...')
                color, row, col = get_pattern_pixel(self.current_pattern, self.pixel_current) 
                if color == 1: self.kolo_direction = 1
                if color == 0: self.kolo_direction = -1
                if color != -1:  # means we still are inside the pattern image                            
                    pos_coverage, dead = check_mapka(self.advanced_mapka, thresh = float(self.extvars['check_threshold']), size = int(self.extvars['pixel_area_size']))
                    
                    # start video recording
                    if self.filename_numa == 0:
                        fname = rmd + self.name_the_file(add = 'timelapse', polar = int(self.rotaround_var.get()) - self.polrot)
                        self.record_timelapse(fname = fname.replace('.npy', '.avi'))
                        
                    
                    if self.filename_numa < int(self.extvars['max_shots']) and dead < float(self.extvars['dead_threshold']):
                        self.burn_hist_gfs.append(pos_coverage)
                        do_kolo = should_try_kolo(self.burn_hist_anims, self.burn_hist_gfs)
                        # if there is not enough of good domian, dont do kolo
                        if (color == 1 and pos_coverage < float(self.extvars['kolo_min_threshold'])) or (color == 0 and pos_coverage > (1-float(self.extvars['kolo_min_threshold']))):
                            do_kolo = False
                        is_finished = False
                        
                        # done with this area
                        if (color == 1 and pos_coverage >= float(self.extvars['kolo_threshold'])) or (color == 0 and pos_coverage <= (1-float(self.extvars['kolo_threshold']))):  # means this pixel is done-done, go to next one
                            self.log('------- autopilot task report: %d pixel is done with %f pos coverage'%(self.pixel_current, pos_coverage))
                            
                            # save info file
                            numa = str(self.filename_numa).zfill(3)
                            fname = rmd + self.name_the_file(add = 'burnParams_'+numa, date=self.savedata_date, polar = 90 - self.polrot)
                            f = open(fname, 'w')
                            f.write('anim_type=%s\n'%('none'))
                            f.write('result=%s\n'%('SUCCESS'))
                            f.write('threshold of %2.4f was reached!'%float(self.extvars['kolo_threshold']))
                            f.close()
                              
                            # end video
                            fname = rmd + self.name_the_file(add = 'timelapse', polar = 90 - self.polrot)
                            self.record_timelapse(fname = fname.replace('.npy', '.avi'))
                            
                            self.pixel_current += 1
                            _, row, col = get_pattern_pixel(self.current_pattern, self.pixel_current) 
                            self.pixel_x = float(self.arealc_x.get())*col
                            self.pixel_y = float(self.arealc_y.get())*row                        
                            self.ccom_index = 0  # repeat full command
                            self.filename_numa = 0
                            self.pixelburn_count = 0
                            self.pixelburns_in_row = 0
                            self.burn_hist_gfs = []
                            self.burn_hist_anims = []
                            self.burn_params_comment = ''
                            is_finished = True
                            
                        # kolopattern should be done
                        if (color == 1 and pos_coverage < float(self.extvars['kolo_threshold']) and pos_coverage >= float(self.extvars['pixel_threshold']))\
                        or (color == 0 and pos_coverage > (1-float(self.extvars['kolo_threshold'])) and pos_coverage <= (1-float(self.extvars['pixel_threshold'])))\
                        or (do_kolo and is_finished == False):  # condition for using kolo instead of pixel if pixel was unsuccesful
                            # set stuff necessary for kolo-animation
                            self.log('------- autopilot task report: calculating validity array...')
                            self.ptact_var.set('shining-moon3')
                            val_arr = new_validity_array(self.advanced_mapka, self.kolo_direction, shot_size = int(self.extvars['shot_size']), thres = float(self.extvars['check_threshold']))
                            self.brush_size = int(self.extvars['shot_size'])
                            self.burn_hist_anims.append('kol')
                            self.paranim_x = val_arr[0, 1]
                            self.paranim_y = val_arr[0, 0]
                            self.paranim_angle = val_arr[0, 2]
                            self.eatspeed_var.set(int(self.extvars['kolo_duration']))
                            
                            self.burn_params_comment = 'comment= do_kolo: %s'%do_kolo
                            
                            self.log('------- autopilot task report: pixel %d: kolo burn will commence for the %d time (total = %d), %2.4f pos coverage'%(self.pixel_current, self.filename_numa-self.pixelburn_count, self.filename_numa, pos_coverage))
                            ld = eval(self.extvars['laser_kolo_function'].replace('N', str(self.filename_numa)))
                            self.mn_set_laserduty(ld)
                            self.filename_numa += 1
                            self.pixelburns_in_row = 0
                            
                        # pixel burn should be done
                        if ((color == 1 and pos_coverage < float(self.extvars['pixel_threshold'])) or (color == 0 and pos_coverage > (1-float(self.extvars['pixel_threshold'])))) \
                        and (do_kolo == False): 
                            # set stuff necessary to run pixelburn
                            self.ptact_var.set('pixel-burn4')
                            self.paranim_x = 224
                            self.paranim_y = 400
                            self.paranim_angle = 0
                            self.brush_size = 300
                            self.burn_hist_anims.append('pix')
                            self.eatspeed_var.set(int(self.extvars['pixel_duration']))
                            self.log('------- autopilot task report: pixel %d: pixel burn will commence for the %d time (total = %d), %2.4f pos coverage'%(self.pixel_current, self.pixelburn_count, self.filename_numa, pos_coverage))
                            ld = eval(self.extvars['laser_function'].replace('N', str(self.filename_numa)))
                            self.mn_set_laserduty(ld)
                            self.filename_numa += 1
                            self.pixelburn_count += 1
                            self.pixelburns_in_row += 1
                            
                            self.burn_params_comment = 'comment= pixelburns (not kolo) in row count: %d, total: %d'%(self.pixelburns_in_row, self.pixelburn_count)
                    else:
                        self.log('------- autopilot task report: %d pixel had been shot %d times, thats enought, not done, sorry (dead factor = %2.4f)!'%(self.pixel_current, self.filename_numa, dead))
                        
                        # save params file
                        numa = str(self.filename_numa).zfill(3)
                        fname = rmd + self.name_the_file(add = 'burnParams_'+numa, date=self.savedata_date, polar = 90 - self.polrot)
                        f = open(fname, 'w')
                        f.write('anim_type=%s\n'%('none'))
                        f.write('result=%s\n'%('FAILED'))
                        f.write('dead_threshold = %2.4f, total tries = %d'%(dead, self.filename_numa))
                        f.close()
                        
                        # end video
                        fname = rmd + self.name_the_file(add = 'timelapse', polar = 90 - self.polrot)
                        self.record_timelapse(fname = fname.replace('.npy', '.avi'))
                        
                        self.pixel_current += 1
                        _, row, col = get_pattern_pixel(self.current_pattern, self.pixel_current) 
                        self.pixel_x = float(self.arealc_x.get())*col
                        self.pixel_y = float(self.arealc_y.get())*row                        
                        self.ccom_index = 0  # repeat full command
                        self.filename_numa = 0
                        self.pixelburn_count = 0
                        self.pixelburns_in_row = 0
                        self.burn_params_comment = ''
                else:
                    self.log('------- autopilot task report: no more pixels left')
                    self.ccom_index = 0
                    self.cur_command_list = [['END']]
                    self.burn_params_comment = ''
            
            
            if self.ctask[0] == 'D_SET_WAIT':  # sets wait timer
                self.wait_timer(self.ctask[1])
                self.ctask_index += 1
                self.log('------- autopilot task report: setting timer to %f'%self.ctask[1])
                
            if self.ctask[0] == 'D_WAIT':  # as in wait for the timer to say its ok to go
                if self.is_wait_done() == True:
                    self.ctask_index += 1
                    self.log('------- autopilot task report: wait is done')
            
            if self.ctask[0] == 'GOTO':
                self.atp_display_map = False
                self.grbl_command('G21 G90 G1 X%2.2f Y%2.2f F%s\n'%(self.ctask[1][0], self.ctask[1][1], self.grbl_speed.get()))
                self.ctask_index += 1
                self.log('------- autopilot task report: initiating grbl repositioning, target X%2.2f Y%2.2f F%s'%(self.ctask[1][0], self.ctask[1][1], self.grbl_speed.get()))
                
            if self.ctask[0] == 'D_FOCUS_STEP':
                self.grbl_command('G21 G91 G1 Z%2.2f F15\n'%(self.ctask[1]))
                self.ctask_index += 1
                self.log('------- autopilot task report: Z-Axis move %2.2f'%(self.ctask[1]))
                
            if self.ctask[0] == 'D_FOCUS_AUTO_DECIDE':
                self.fcs = focus_check(self.afocus_pos, self.afocus_fcs)
                if self.fcs == None:  
                    self.ctask_index += 1
                    self.afocus_round += 1
                    self.ccom_index -= 1
                    self.log('------- autopilot task report: repeating autofocus command')
                
                if self.afocus_round > 6 or self.fcs != None:  
                    self.ctask_index += 1
                    
                    self.ccom_index -= 1
                    self.afocus_step = -0.5
                    
                    self.log('------- autopilot task report: autofocus is done')
                    
                    f = open('temp2/pos.txt', 'w')
                    for p in self.afocus_pos:
                        f.write('%f\n'%p)
                    f.close()
                    f = open('temp2/fcs.txt', 'w')
                    for p in self.afocus_fcs:
                        f.write('%f\n'%p)
                    f.close()
                    
            if self.ctask[0] == 'D_AFOCUS_APPEND':
                self.afocus_pos.append(self.ctask[1])
                self.afocus_fcs.append(cv2.Laplacian(frame, cv2.CV_64F).var())                
                self.ctask_index += 1
                self.log('------- autopilot task report: appending focus to list')
            
            if self.ctask[0] == 'D_SETPOL':
                self.rotate_elli(self.ctask[1])
                self.ctask_index += 1
                self.log('------- autopilot task report: setting polarizator to %f'%self.ctask[1])
                
            if self.ctask[0] == 'D_SETPOL_TOP':
                self.rotate_elli_top(self.ctask[1])
                self.log('------- autopilot task report: setting top polarizator to %f'%self.ctask[1])
                    
                self.ctask_index += 1
                
            if self.ctask[0] == 'D_IN_MIRROR':
                self.btnAct_ell6_in()
                self.ctask_index += 1
                self.log('------- autopilot task report: mirror in!')
                
            if self.ctask[0] == 'D_OUT_MIRROR':
                self.btnAct_ell6_out()
                self.ctask_index += 1
                self.log('------- autopilot task report: mirror out!')
                
            if self.ctask[0] == 'D_LASER_STATE':
                if self.laserstate == 'OFF' and self.ctask[1] == 1:
                    self.btn_laser_switch()
                if self.laserstate == 'ON' and self.ctask[1] == 0:
                    self.btn_laser_switch()
                self.ctask_index += 1
                self.log('------- autopilot task report: switching laser to %s state'%['OFF', 'ON'][self.ctask[1]])
                
            if self.ctask[0] == 'D_SAVEFRAME':
                save_path = rmd
                # save_path = data_dir + 'raw_mikro_data/'
                save_path = save_path + self.name_the_file(add='frame_'+self.ctask[1], date=self.savedata_date, polar = int(self.rotaround_var.get()) - self.polrot)
                self.blinker_multip = 0
                np.save(save_path, frame)
                self.ctask_index += 1
                self.log('------- autopilot task report: saving current frame as %s'%save_path)
                
            if self.ctask[0] == 'D_SAVEFRAME_OFFSETS':
                save_path = rmd
                # save_path = data_dir + 'raw_mikro_data/'
                save_path = save_path + 'offsets_frames/frame_for_offsets_POL%d.npy'%self.ctask[1]
                self.blinker_multip = 0
                np.save(save_path, frame)
                self.ctask_index += 1
                self.log('------- autopilot task report: saving current frame as %s'%save_path)
                
            if self.ctask[0] == 'D_SAVEMAP':  # obsolete? NO!
                save_path = rmd
                # save_path = data_dir + 'raw_mikro_data/'
                if self.ctask[1] == 'numa':
                    numa = str(self.filename_numa).zfill(3)
                    np.save(save_path + self.name_the_file(add = 'mapka_'+numa, date=self.savedata_date, polar = int(self.rotaround_var.get()) - self.polrot), self.advanced_mapka)
                else:                    
                    np.save(save_path + self.name_the_file(add = 'mapka_'+self.ctask[1], date=self.savedata_date, polar = int(self.rotaround_var.get()) - self.polrot), self.advanced_mapka)
                self.ctask_index += 1
                self.log('------- autopilot task report: saving current mapka as %s'%save_path)
                
            if self.ctask[0] == 'D_CURDATE':
                self.savedata_date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                self.ctask_index += 1
                self.log('------- autopilot task report: setting savedata_date to %s'%self.savedata_date)
                
            if self.ctask[0] == 'D_SAVEDATA':
                save_path = rmd
                # save_path = data_dir + 'raw_mikro_data/'
                if self.ctask[1] == 'numa':
                    numa = str(self.filename_numa).zfill(3)
                    np.save(save_path + self.name_the_file(add='frame_'+numa, date=self.savedata_date, polar = 90 - self.polrot), self.amap_frameM)
                    np.save(save_path + self.name_the_file(add='frame_'+numa, date=self.savedata_date, polar = 90 + self.polrot), self.amap_frameP)
                else:
                    np.save(save_path + self.name_the_file(add='frame_'+self.ctask[1], date=self.savedata_date, polar = 90 - self.polrot), self.amap_frameM)
                    np.save(save_path + self.name_the_file(add='frame_'+self.ctask[1], date=self.savedata_date, polar = 90 + self.polrot), self.amap_frameP)
                # np.save(save_path + self.name_the_file(add='baseline_'+self.ctask[1], date=self.savedata_date, polar = 90 - self.polrot), self.baselineM)
                # np.save(save_path + self.name_the_file(add='baseline_'+self.ctask[1], date=self.savedata_date, polar = 90 + self.polrot), self.baselineP)
                self.ctask_index += 1
                self.log('------- autopilot task report: saving current mapdata in %s'%save_path)
                
            if self.ctask[0] == 'REMFRAME':
                if self.ctask[1] == 'M': self.amap_frameM = frame.copy()
                if self.ctask[1] == 'P': self.amap_frameP = frame.copy()
                self.ctask_index += 1
                self.log('------- autopilot task report: saving frame %s to internal memory'%self.ctask[1])
                
            if self.ctask[0] == 'CALCMAP':
                self.make_rchannel_map()
                self.ctask_index += 1
                self.atp_display_map = True
                self.log('------- autopilot task report: calculating rchannel map')
                
            if self.ctask[0] == 'D_ILLUMINATE':
                if self.ctask[1] == 0:
                    if self.laserstate == 'ON': self.btn_laser_switch()
                if self.ctask[1] > 0:
                    self.mn_set_laserduty(self.ctask[1])
                    if self.laserstate == 'OFF': self.btn_laser_switch()
                
                if self.raw_projector_screen == True:
                    self.btn_raw_play()
                
                self.ctask_index += 1
                self.log('------- autopilot task report: laser illumination set to %2.4f'%self.ctask[1])
                
            if self.ctask[0] == 'D_POLARLIST':
                # directly rotate elliptec by angle from list as absolute position
                angle = self.crosspolarize_angle_list[self.crp_pos]
                command = angle_to_ellocommand(angle)
                self.elliptec.write(command)
                self.ctask_index += 1
                self.log('------- autopilot task report: rotate elliptec to absoute position of %d'%angle)
                
            if self.ctask[0] == 'D_RAW_WAIT_END':
                # wait for end of animation (raw proj output) to turn off the laser
                if self.raw_proj_counter == len(self.raw_proj_imgs):
                    if self.laserstate == 'ON': self.btn_laser_switch()
                    self.ctask_index += 1
                    self.log('------- autopilot task report: finished animation, laser state off')
                
            if self.ctask[0] == 'D_MEASURE_CRP_INT':
                # measures total intensity of light and saves to file
                suma = frame.sum()
                angle = self.crosspolarize_angle_list[self.crp_pos]
                self.crp_pos += 1
                self.ctask_index += 1
                self.log('------- autopilot task report: measured instesity for angle %d: %d'%(angle, suma))
                
            if self.ctask[0] == 'D_POLARLIST_CHECK':
                if self.crp_pos < len(self.crosspolarize_angle_list):
                    self.ccom_index += self.ctask[1]
                self.ctask_index += 1
                self.log('------- autopilot task report: changing ccom index by %d (now is %d)'%(self.ctask[1], self.ccom_index))
            
            if self.ctask[0] == 'CALC_BMAP':
                # finding last baseline with current aquisition parameters   
                date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                baseline_params = 'P%2.2f G%d E%d %s'%(self.polrot, self.camera_gain, self.camera_expo, self.current_obiektyw)
                self.baselineM, self.baselineP, time2 = self.read_last_baseline(baseline_params)
                
                if time2 != -1:            
                    # a_im_p = (cv2.cvtColor(self.amap_frameP, cv2.COLOR_BGR2GRAY)/255).astype(float)
                    # a_im_m = (cv2.cvtColor(self.amap_frameM, cv2.COLOR_BGR2GRAY)/255).astype(float)
                    params = self.advMapOptions.get()
                    params = params.replace(' ', '')
                    params += ','
                    # np.save(data_dir + 'raw_mikro_data/' + '%s mapka_img_p.npy'%date, self.amap_frameP)
                    # np.save(data_dir + 'raw_mikro_data/' + '%s mapka_img_m.npy'%date, self.amap_frameM)
                    # np.save(data_dir + 'raw_mikro_data/' + '%s mapka_bas_p.npy'%date, baselineP)
                    # np.save(data_dir + 'raw_mikro_data/' + '%s mapka_bas_m.npy'%date, baselineM)
                    sg = int(get_amap_parameter(params, 'G'))
                    ch = int(get_amap_parameter(params, 'CH'))
                    bas = int(get_amap_parameter(params, 'B'))
                    self.advanced_mapka = baseline_mapping(self.amap_frameM, self.amap_frameP, self.baselineM*bas, self.baselineP*bas, sigma = sg, ch = ch)
                    self.amap_ranger = 0.07
                    self.atp_display_map = True
                    self.log('------- autopilot task report: calculating bmap using %s, acquired %d second ago'%(baseline_params, time2))
                else:
                    self.log('------- autopilot task report: no baseline found for parameters %s'%baseline_params)
                self.ctask_index += 1
                
            if self.ctask[0] == 'CALC_BASELINE':
                # this actually saves both plus and minus images
                
                date = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                baseline_nameP = '%s baselineP P%2.2f G%d E%d %s'%(date, self.polrot, self.camera_gain, self.camera_expo, self.current_obiektyw)
                baseline_nameM = '%s baselineM P%2.2f G%d E%d %s'%(date, self.polrot, self.camera_gain, self.camera_expo, self.current_obiektyw)
                np.save(data_dir + 'raw_mikro_data/baselines/' + baseline_nameP, self.amap_frameP)
                np.save(data_dir + 'raw_mikro_data/baselines/' + baseline_nameM, self.amap_frameM)
                
                self.ctask_index += 1
                self.atp_display_map = True
                self.log('------- autopilot task report: saving baseline as %s'%baseline_nameP)
                
            if self.ctask[0] == 'D_CALC_SHOT_ARR':
                self.shot_array = new_validity_array(self.advanced_mapka, self.ctask[1])
                # save shot array
                if self.ctask[1] != 0:
                    save_path = self.name_the_file(add = 'shot_arr', date = self.savedata_date)
                    np.save(rmd + save_path, self.shot_array)
                    # np.save(data_dir + 'raw_mikro_data/' + save_path, self.shot_array)
                    
                    # additionally save all relevant data
                    rel_data_path = self.name_the_file(add = 'relevant_data', date = self.savedata_date)
                    # f = open(data_dir + 'raw_mikro_data/' + rel_data_path.replace('.npy', '.txt'), 'w')
                    f = open(rmd + rel_data_path.replace('.npy', '.txt'), 'w')
                    f.write('eatspeed = %f\n'%float(self.eatspeed_var.get()))
                    f.write('temperature = %f\n'%self.t_cur)
                    f.close()                    
                else:
                    save_path = 'not saved actually...'
                self.ctask_index += 1
                self.log('------- autopilot task report: shot_array calculated and saved as %s'%save_path)
                
            if self.ctask[0] == 'FLIP':
                self.brush_size = 110
                self.interaction_mode = 'point'
                self.pt_point = (int(self.shot_array[0][v_shot_x]), int(self.shot_array[0][v_shot_y]))
                time_wait = float(self.eatwait_var.get())+1
                self.pt_eat_stage = -time_wait*float(self.eatspeed_var.get())/100
                self.pt_eat_dir = 180+self.shot_array[0][v_angle]
                self.ctask_index += 1
                self.atp_display_map = False
                self.log('------- autopilot task report: initiate flipping domains...')
                
            if self.ctask[0] == 'D_SET_TEMP':
                # sets temperature
                self.set_temp_var.set(str(self.ctask[1]))
                self.set_temperature()
                
                self.ctask_index += 1
                self.log('------- autopilot task report: setting temperature to %2.1f'%self.ctask[1])
                
            if self.ctask[0] == 'D_WAIT_TEMP':
                set_temp = float(self.set_temp_var.get())
                diff = abs(self.thortemp_Tactual - set_temp)
                if diff <= self.ctask[1]:
                    self.ctask_index += 1
                    self.log('------- autopilot task report: designated temperature reached')
                    
            if self.ctask[0] == 'D_WAIT_PARANIM':
                if self.camera_point_draw.max() == 0:
                    self.ctask_index += 1
                    self.log('------- autopilot task report: paranim finished!')
                
            if self.ctask[0] == 'WAIT_FLIP':
                if self.pt_eat_stage >= 1:
                    self.ctask_index += 1
                    self.log('------- autopilot task report: flipping finished')
                    
            if self.ctask[0] == 'D_CCOM_INDEX':
                self.ccom_index += self.ctask[1]
                self.ctask_index += 1
                self.log('------- autopilot task report: changing ccom index by %d (now is %d)'%(self.ctask[1], self.ccom_index))
                
            if self.ctask[0] == 'D_SET_CCOM_INDEX':
                self.ccom_index = self.ctask[1]
                self.ctask_index += 1
                self.log('------- autopilot task report: setting ccom index to %d'%(self.ccom_index))
                
            if self.ctask[0] == 'D_EVALUATE_MAP':
                n, p = estimate_total(self.advanced_mapka)
                self.shot_cnt += 1
                if int(self.eatdir_var.get()) == 1:
                    est = n/(n+p)
                else:
                    est = p/(n+p)
                    
                if est < 0.25:
                    ccom_increment = self.ctask[1]
                    self.shot_cnt = 0
                else:
                    ccom_increment = self.ctask[2]
                    
                if self.aclear == True or self.shot_cnt >= 3:
                    self.shot_cnt = 0
                    ccom_increment = self.ctask[1]
                
                self.ccom_index += ccom_increment
                self.ctask_index += 1
                self.log('------- autopilot task report: map evaluation result %f (aclear = %d), meaning increment is %d'%(est, self.aclear, ccom_increment))
                self.aclear = False
                
            if self.ctask[0] == 'D_COMBINE_MAPS':
                params = self.advMapOptions.get()
                params = params.replace(' ', '')
                params += ','
                sg = int(get_amap_parameter(params, 'G'))
                ch = int(get_amap_parameter(params, 'CH'))
                bas = int(get_amap_parameter(params, 'B'))
                
                combine_by_date(self.savedata_date, obj = self.current_obiektyw, sname = self.path.get(), sigma = sg, ch = ch)
                self.ctask_index += 1
                self.log('------- autopilot task report: combining map!')
                            
            if self.ctask[0] == 'D_MMAP_ADD_FRAME_M':
                img = translate_image(frame, -self.ctask[1][0], -self.ctask[1][1]) 
                # img = frame
                    
                self.mmap_minus += img.astype(float)
                # self.blinker_multip = 0
                self.ctask_index += 1
                self.log('------- autopilot task report: +added frame to mm_minus!')
                
            if self.ctask[0] == 'D_MMAP_ADD_FRAME_P':
                img = translate_image(frame, -self.ctask[1][0], -self.ctask[1][1]) 
                # img = frame
                    
                self.mmap_plus += img.astype(float)
                # self.blinker_multip = 0
                self.ctask_index += 1
                self.log('------- autopilot task report: +added frame to mm_plus!')
                
            if self.ctask[0] == 'D_MMAP_MAKE':
                angles = [int(a) for a in self.map_rots_var.get().replace(' ', '').split(',')]
                divider = len(angles)*self.mm_count
                    
                self.mmap_plus /= divider
                self.mmap_minus /= divider
                
                self.amap_frameP = np.round(self.mmap_plus).astype(np.uint8)
                self.amap_frameM = np.round(self.mmap_minus).astype(np.uint8)
                
                params = self.advMapOptions.get()
                params = params.replace(' ', '')
                params += ','
                sg = int(get_amap_parameter(params, 'G'))
                ch = int(get_amap_parameter(params, 'CH'))
                bas = int(get_amap_parameter(params, 'B'))
                
                self.advanced_mapka = baseline_mapping(self.amap_frameP, self.amap_frameM, self.amap_frameP*0, self.amap_frameP*0, sigma = sg, ch = ch)
                self.amap_ranger = 0.07
                self.atp_display_map = True
                
                self.ctask_index += 1
                self.log('------- autopilot task report: calculating mmap!')
                
            if self.ctask[0] == 'D_START_PARANIM':
                # starts the paranim
                self.interaction_mode = 'parametric_animation'
                self.paranim_start_time = time.time()  # this inits the animation
                self.atp_display_map = False  # do not display map but real-time image
                
                self.ctask_index += 1
                self.log('------- autopilot task report: initiate parametric animation!')
                
            if self.ctask[0] == 'D_CHANGE_GAIN':
                # it is a little weird but it works (or it doesnt?) (it does - function deltagain is used to change gain about delta, not set it to delta) (for new camera, expo is set)
                # self.camera_gain = self.ctask[1]
                # self.camera_deltagain(0)
                self.camera_setexpo(self.ctask[1])
                
                self.ctask_index += 1
                self.log('------- autopilot task report: changing camera gain to %d'%self.camera_gain)
                
            if self.ctask[0] == 'D_RANDOMIZE_PARANIM':
                # get random variables
                self.paranim_variables = pixelburn2_rnd_variables()
                params_str = 'v0=%2.1f, v1=%2.1f, v2=%2.1f, v3=%2.1f, v4=%2.1f'%(self.paranim_variables[0], self.paranim_variables[1], self.paranim_variables[2], self.paranim_variables[3], self.paranim_variables[4])   
                self.paranim_var_str.set(params_str)                
                
                self.paranim_length_ms = 1000*int(rnd_between(30, 60))
                self.eatspeed_var.set(str(self.paranim_length_ms/1000))
                
                self.brush_size = 290
                
                # self.laserpower = 5+0.25*int(50/0.25*np.random.random())
                self.laserpower = int(rnd_between(12, 22))
                self.mn_set_laserduty(self.laserpower)
                                
                # temp = 65 + int(25*random.random())
                # self.set_temp_var.set(str(temp))
                
                self.ctask_index += 1
                self.log('------- autopilot task report: parameters randomized')
                
            if self.ctask[0] == 'NEXTPOS':
                self.atp_display_map = False
                self.ask_grbl_position()
                self.cpos += 1
                self.ctask_index += 1
                x, y = self.position_list[self.cpos]
                t = move_time([self.grblX, self.grblY], [x, y], float(self.grbl_speed.get())) + 1
                self.wait_timer(t)
                self.grbl_command('G21 G90 G1 X%2.2f Y%2.2f F%s\n'%(x, y, self.grbl_speed.get()))
                self.log('------- autopilot task report: initiating grbl repositioning, target X%2.2f Y%2.2f F%s'%(x, y, self.grbl_speed.get()))
                

                    
        
        # from now, self.cv2image is what will be displayed        
        # overlay camera_draw as red channel with selected alpha if hold is on
        if self.always_overlay > 0:
            try:
            # overlay draw image
                zer = np.zeros((self.camera_draw.shape[0], self.camera_draw.shape[1])).astype(np.uint8)
                red_overlay = cv2.merge((zer, zer, self.camera_draw[:, :, 0]))                
                self.cv2image = cv2.addWeighted(self.cv2image, 1, red_overlay, 0.3, 0.0)
                
                # overlay point draw image
                zer = np.zeros((self.camera_point_draw.shape[0], self.camera_point_draw.shape[1])).astype(np.uint8)
                red_overlay = cv2.merge((zer, zer, self.camera_point_draw[:, :, 0]))
                self.cv2image = cv2.addWeighted(self.cv2image, 1, red_overlay, 0.3, 0.0)
            except:
                print('red_overlay shape:')
                print(red_overlay.shape)
                print('cv2image shape:')
                print(self.cv2image.shape)
            

        # draw brush circle
        if self.interaction_mode == 'draw':
            self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), self.brush_size, (220, 80, 80), 2)
        
        if self.interaction_mode == 'point' or self.interaction_mode == 'parametric_animation':
            if self.B1_was_pressed == True:  # means mouse was already pressed, draw circle and arrow at place where it was pressed
                # draw stationary circle
                self.cv2image = cv2.circle(self.cv2image, (self.B1_pressed_at[0], self.B1_pressed_at[1]), self.brush_size, (220, 80, 80), 1)
                self.cv2image = cv2.circle(self.cv2image, (self.B1_pressed_at[0], self.B1_pressed_at[1]), 5, (220, 80, 80), -1)
                
                # draw nice arrow indicating eat direction
                dx = self.mouse_x - self.B1_pressed_at[0]+0.00001
                dy = self.mouse_y -76+50 - self.B1_pressed_at[1]
                alfa = np.arctan(dy/dx)
                if (dx<0): alfa += np.pi
                dx = int(self.brush_size*np.cos(alfa))
                dy = int(self.brush_size*np.sin(alfa))
                self.cv2image = cv2.arrowedLine(self.cv2image, self.B1_pressed_at, (self.B1_pressed_at[0]+dx, self.B1_pressed_at[1]+dy), (255, 255, 255), 1)
            else:  # mouse is not pressed, draw circe at current location        
                self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), self.brush_size, (220, 80, 80), 1)
                self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), 5, (220, 80, 80), -1)
  
           
        # blinker action
        if self.blinker_multip < 1:
            self.cv2image = (self.cv2image*self.blinker_multip).astype(np.uint8)
            self.blinker_multip += 0.1
            
        # adds infobar (if not already present, for example when camera_image_type == 'MAP', cv2image does not reset every loop)
        if self.cv2image.shape[0] == 448:
            self.cv2image = self.add_infobar(self.cv2image)
        
        
        # cv2image = cv2.flip(cv2image, 0)
        img = Image.fromarray(self.cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        # self.lmain.imgtk = imgtk
        # self.lmain.configure(image=imgtk)
        # self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.imgtk, anchor=CENTER)
        
        # record video
        if self.video_writer != None:
            if self.frame_lapse_counter >= self.frame_lapse:
                self.video_writer.write(cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB))
                # print("show frame ", cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB).shape)
                self.frame_lapse_counter = 0
            else:
                self.frame_lapse_counter += 1
        
        # save image
        if self.save_image == True:
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(data_dir + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.png'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    img.save(data_dir + 'saved_images/' + fname + '_%s.png'%str(num).zfill(2))
                    self.log(fname + '_%s.png saved!'%str(num).zfill(2))
                    saved =1
                    
            self.save_image = False
            
            
        # save raw frame data
        if self.save_data == True and frame_is_new == True:
            fname = self.path.get()
            dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
            fname = dt_string + ' ' + fname
            path_name = rmd + fname
            np.save(path_name, frame)
            self.save_data = False
           
        # record raw frame (basically save frame but with specified freqency)
        if self.rec_data == True:
            if self.rec_data_cnt >= int(self.savedata_freq_var.get()):
                self.rec_data_cnt = 0
                fname = self.path.get()
                dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                fname = dt_string + ' ' + fname + ' (auto)'
                path_name = rmd + fname
                np.save(path_name, frame)
            self.rec_data_cnt += 1


    def add_infobar(self, cv2image):
        # params
        # scalebar_x = 600
        scalebar_y = 10
        scb_h = 20
        
        # create empty panel (with black line at top)
        parameters_panel = np.zeros((bar_height, 800, 3), np.uint8)
        parameters_panel[:,:] = (220, 220, 220)
        parameters_panel[:3, :] = (40, 40, 40)
        
        # add scale bar        
        scalebar_x = 800-scale_px[self.current_obiektyw]-20
        parameters_panel = cv2.rectangle(parameters_panel, (scalebar_x, scalebar_y), (scalebar_x+scale_px[self.current_obiektyw], scalebar_y+scb_h), (40,40,60), -1)
        text = '%s um'%scale_um[self.current_obiektyw]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # add scale text
        textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
        textX = int((scalebar_x+scale_px[self.current_obiektyw]/2) - (textsize[0] / 2))
        textY = int(scalebar_y+scb_h/2 + (textsize[1] / 2))        
        parameters_panel = cv2.putText(parameters_panel, text, (textX, textY), font , 0.5, (240, 240, 250), 1, cv2.LINE_AA)

        # add text (using pillow)
        text = ''
        if self.thortemp_Tactual > 0:
            text = 'temperature = %2.2f °C\n'%self.thortemp_Tactual
        else:
            text = 'temperature = sensor error\n'
        text += 'polarizators (T|B) = %2.2f | %2.2f\n'%(float(self.ell_var2_top.get()), float(self.ell_var2.get()))
        text += 'objective = %s\n'%self.current_obiektyw
        text += 'bright = %.2f\n'%self.brightness
        if self.atp_display_map == True:
            text += 'map polars rotations: %s\n'%(self.map_rots_var.get())
        # text += 'focus = %f'%cv2.Laplacian(cv2image, cv2.CV_64F).var()
        img_pil = Image.fromarray(parameters_panel)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), text, font = self.font11, fill = (10, 10, 10))
        parameters_panel = np.array(img_pil)
        
        
        # merge with camera image (cv2image)
        new_img = np.zeros((100+448, 800, 3), np.uint8)
        new_img[:448, :] = cv2image
        new_img[448:, :] = parameters_panel
        return new_img


    def mod_source(self, value):
        global cap
        cap.release()
        cap = cv2.VideoCapture(int(value))
        self.log(value)
        
        
    def exit(self):
        self.log('exit!')
        
        self.event.set()
        
        self.log('initiating close of cam_reader process')
        if self.elliptec != None:
            self.rotate_elli(90)
            # self.elliptec.write(command)
        # if self.elliptec_top != None:
            # self.rotate_elli(90)
        root.quit()
        
       
    def initiate_projector_window(self):
        if self.projector_window == None:                
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = Toplevel(root)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            
    
    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
    
        
    def activate_projector_window(self):
        self.log('Projector window activated!')
        
        # initialize full screen mode
        self.projector_window.overrideredirect(True)
        self.projector_window.state("zoomed")
        # self.projector_window.activate()
        
        self.canvas_proj = Canvas(self.projector_window, width=1024, height=768, bg='black', highlightthickness=0, relief='ridge')
        self.canvas_proj.pack(side = LEFT)
        
        self.log('Loading projector calibration...')
        self.load_calib()
    
        
    def refresh_projector_image(self):
        try:
            if self.interaction_mode == 'draw':  # if drawing is on use camera draw and transform it into projector array
                im_out = cv2.warpPerspective(self.camera_draw, self.homomatrix, (1024, 768))
                self.projector_arr = im_out
            if self.interaction_mode == 'point' or self.interaction_mode == 'parametric_animation' or self.interaction_mode == 'image':
                im_out = cv2.warpPerspective(self.camera_point_draw, self.homomatrix, (1024, 768))
                self.projector_arr = im_out
            if self.interaction_mode == 'none':  # for now, just put an eagle there (maybe it should be black screen?)
                if projector_calib_c >= calib_dots_dim**2:  # if it is not calibrating
                    self.projector_arr = self.default_image
            
            if self.hold_proj == 0:
                # img = Image.fromarray(cv2.flip(add_border(self.projector_arr, self.border), 1))
                self.projector_arr_border = add_border(self.projector_arr, self.border)
                img = Image.fromarray(cv2.flip(self.projector_arr_border, 1))
            else:
                img = Image.fromarray(np.zeros((768, 1024, 3), np.uint8))
            
            # this is to display raw image on projector without border, without anything
            if self.raw_projector_screen == True:
                # img = Image.fromarray(self.raw_proj_curr_img)
                # img = Image.fromarray(np.zeros((768, 1024, 3), np.uint8))
                self.projector_arr = self.raw_proj_curr_img
                img = Image.fromarray(self.projector_arr)
            
            
            
            # img = Image.fromarray(np.zeros((768, 1024, 3), np.uint8))
            
            
            self.proj_imgtk = ImageTk.PhotoImage(image=img)
            self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=CENTER)
        except:
            pass
        
        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.


if __name__ == '__main__':
    root = Tk()
    # root.geometry("1920x1080")

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    root.state("zoomed")
    # root.iconbitmap('bicon02.ico')

    #creation of an instance
    app = Window(root)
    # app.show_frame()

    #mainloop
    root.iconbitmap('laser.ico')
    root.mainloop()
        
        
        
        
        
        
        
        

# save_image = False

# root = Tk()
# root.bind('<Escape>', lambda e: exit())

# source_frame = Frame(root)
# source_frame.grid(row=0, column=0, padx = padd)

# sv = StringVar()
# source_entry = Entry(source_frame, textvariable=sv)
# source_entry.pack()

# buttonSource = Button(source_frame, text = 'Change source', command = mod_source)
# buttonSource.pack()


# lmain = Label(root)
# lmain.grid(row=1, column=0, padx = padd)

# buttonSave = Button(root, text = 'Save', command = save_img)
# buttonSave.grid(row=2, column=0, padx = padd)

# path = StringVar()
# nameEntered = Entry(root, width = 15, textvariable = path)
# nameEntered.grid(row=3, column=0, padx = padd)

# show_frame()
# root.mainloop()