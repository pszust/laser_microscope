
import cv2
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
    img_min = np.mean(imgs_min, axis=0).astype(np.uint8)
    img_pls = np.mean(imgs_pls, axis=0).astype(np.uint8)

    if ch != -1:
        img_min_g = dechannel(img_min.copy(), ch)
        img_pls_g = dechannel(img_pls.copy(), ch)
    else:
        img_min_g = img_min.copy()
        img_pls_g = img_pls.copy()
    
    img_min_g = (cv2.cvtColor(img_min_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    img_pls_g = (cv2.cvtColor(img_pls_g, cv2.COLOR_BGR2GRAY)).astype(float)/255
    
    for i in range(0, 2):
        img_min_g = gaussian_filter(img_min_g, sigma=sigma)
        img_pls_g = gaussian_filter(img_pls_g, sigma=sigma)
    return img_pls_g-img_min_g