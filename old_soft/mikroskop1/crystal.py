import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2
import os
import datetime as dt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# import pytesseract
from PIL import Image
import re


def plot_images(
    images, size=10, rowsplit=4, names=None, cmap="gray", axes="on", rng=None
):
    # change the figure size
    ncols = min(len(images), rowsplit)
    nrows = (len(images) - 1) // rowsplit + 1
    width = min(len(images) * size, rowsplit * size)
    height = ((len(images) - 1) // rowsplit + 1) * size
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(width / 2.54, height / 2.54)

    if len(images) == 1:
        # one image, axs is really ax
        if rng:
            axs.imshow(
                images[0], interpolation="none", cmap=cmap, vmin=rng[0], vmax=rng[1]
            )
        else:
            axs.imshow(images[0], interpolation="none", cmap=cmap)
    else:
        # create as many plots as there is space (nrows*ncols)
        for i in range(ncols * nrows):
            if nrows == 1:  # if only one row is present, axs is indexed by one index
                ax = axs[i % rowsplit]
            else:
                ax = axs[i // rowsplit, i % rowsplit]

            # fill axes with images
            if i < len(images):
                image = images[i]
                if rng:
                    ax.imshow(
                        image, interpolation="none", cmap=cmap, vmin=rng[0], vmax=rng[1]
                    )
                else:
                    ax.imshow(image, interpolation="none", cmap=cmap)
                if axes == "off":
                    ax.set_axis_off()
                    # plt.axis('off')
                    # ax = plt.gca()
                    # ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
                    # ax.yaxis.set_major_locator(mpl.ticker.NullLocator())

                # add ax title if provided
                if names:
                    ax.set_title(names[i])
            else:
                # fill blank axs with blank white image
                ax.set_axis_off()
                blank = np.ones((10, 10, 3)) * 255
                ax.imshow(blank, interpolation="none", cmap=cmap)
                # ax = plt.gca()
                # ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
                # ax.yaxis.set_major_locator(mpl.ticker.NullLocator())

    plt.tight_layout()
    plt.close()
    return fig


def cut_frame(frame):
    return frame[0:448, 100:698]


def read_video(path, cut=True, nth=1):
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if cut == True:
                frame = cut_frame(frame)
            # h, w = frame.shape[:2]
            # frame = cv2.resize(frame, (2*w, 2*h))
            if i % nth == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
        i += 1
    cap.release()
    return frames


def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def threshold_crystals(image, threshold=150, blurry=5):
    # sampleImg[sampleImg>180] = 255
    # imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgGray = image[:, :, 1]
    blur = cv2.medianBlur(imgGray, blurry)

    # use adaptive gaussian thresholding (other also avalible)
    blocksize = 251
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2
    )
    _, thr = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    thr = np.where(thr == 255, 0, 255).astype(np.uint8)

    # plot_images([img, blur, thr])
    return thr


def get_contours(threshold, circ=0.2, areas=[30, 6000]):
    if cv2.__version__ == "4.6.0":
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        _, contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    filteredContours = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        reject = 0
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < circ:
            reject = 1

        if area < areas[0] or area > areas[1]:
            reject = 1

        center = cv2.minAreaRect(cnt)[0]
        if threshold[int(center[1]), int(center[0])] == 0:
            reject = 1

        if reject == 0:
            filteredContours.append(cnt)

    rects = [cv2.minAreaRect(cnt) for cnt in filteredContours]
    boxes = [np.int0(cv2.boxPoints(rect)) for rect in rects]
    return filteredContours, rects, boxes


def get_current_crystals(img, minSize=200, threshold = 150, blur = 9, minArea = 30, maxArea = 32_000):
    thr = threshold_crystals(img, blurry=blur, threshold = threshold)
    contours, rects, boxes = get_contours(thr, circ=0, areas=[minArea, maxArea])
    currentCrystals = [
        [cnt, rect]
        for cnt, rect in zip(contours, rects)
        if rect[1][0] * rect[1][1] > minSize
    ]
    return currentCrystals


def crystal_match(crystal1, crystal2):
    cnt1, rect1 = crystal1
    cnt2, rect2 = crystal2
    physicalDst = point_distance(rect1[0], rect2[0])  # distnace in pixels
    sizeDiff = abs(
        rect1[1][0] * rect1[1][1] - rect2[1][0] * rect2[1][1]
    )  # size difference in px**2
    return physicalDst + sizeDiff * 0.1


def update_tracking(tracked, currentCrystals):
    updatedTracked = []
    for trackedCrystal in tracked:
        if trackedCrystal:  # means the crystal is not yet lost
            distances = [
                crystal_match(trackedCrystal, crystal) for crystal in currentCrystals
            ]
            match = np.argmin(distances)
            if distances[match] < 100:
                updatedTracked.append(currentCrystals[match])
            else:
                updatedTracked.append(None)  # lost tracking of crystal
                print("Lost tracking of crystal!")
        else:
            updatedTracked.append(None)
    return updatedTracked


# functions for laser illumination
def paste_centrally(canvas, pattern):
    # Calculate starting position to paste the pattern in the center of the canvas
    start_x = (canvas.shape[1] - pattern.shape[1]) // 2
    start_y = (canvas.shape[0] - pattern.shape[0]) // 2

    # Calculate ending position of the pattern
    end_x = start_x + pattern.shape[1]
    end_y = start_y + pattern.shape[0]

    # Paste the pattern into the canvas
    canvas[start_y:end_y, start_x:end_x] = pattern

    return canvas


def normalize_crystal(crystal):
    """returns normalized contour, i.e. with center at 0, 0"""
    cnt, rect = crystal
    x, y = int(rect[0][0]), int(rect[0][1])
    newCnt = cnt.copy()
    newCnt[:, 0, 0] -= x
    newCnt[:, 0, 1] -= y
    return newCnt


def rotate_contour(cntNorm, angle):
    """angle in deg"""
    xCoords = cntNorm[:, :, 0]
    yCoords = cntNorm[:, :, 1]
    xRot, yRot = rotate_coords(xCoords, yCoords, angle)
    rotated = np.concatenate(
        (xRot[:, :, np.newaxis], yRot[:, :, np.newaxis]), axis=2
    ).astype(int)
    return rotated


def rotate_coords(xCoords, yCoords, angle):
    """angle in deg"""
    cosAlfa = np.cos(angle * np.pi / 180)
    sinAlfa = np.sin(angle * np.pi / 180)
    xRot = xCoords * cosAlfa - yCoords * sinAlfa
    yRot = xCoords * sinAlfa + yCoords * cosAlfa
    return xRot, yRot


def step_toward(pos, center=[200, 200]):
    x, y = pos
    doubleStep = 0
    if x > center[0]:
        x -= 1
        doubleStep += 1
    if x < center[0]:
        x += 1
        doubleStep += 1
    if y > center[1]:
        y -= 1
        doubleStep += 1
    if y < center[1]:
        y += 1
        doubleStep += 1
    if doubleStep == 2:
        return x, y, 1
    else:
        return x, y, 0


def check_angle(pattern, crystal, angle, ret_images=False, expl=False):
    canva = (
        paste_centrally(np.zeros((400, 400), dtype=np.uint8), pattern) / 255
    ).astype(int)

    # rotating contour
    cntNorm = normalize_crystal(crystal.copy())
    rotated = rotate_contour(cntNorm, angle)

    cntDraw = rotated + 200  # centerize
    contourDrawed = cv2.drawContours(
        np.zeros((400, 400)), [cntDraw], -1, (1), -1
    ).astype(int)
    diff = contourDrawed - canva

    distances = []
    badCount = 0
    for x in range(diff.shape[1]):
        for y in range(diff.shape[0]):
            if diff[y, x] == -1:
                cx, cy = x, y
                dst = 0
                while (contourDrawed[cy, cx]) == 0:
                    cx, cy, doubleStep = step_toward([cx, cy])
                    if doubleStep:
                        dst += 1.41
                    else:
                        dst += 1
                    if dst > 1000:
                        print("shit", cx, cy)
                        break
                distances.append(dst)
                if badCount < max(distances):
                    badCount = max(distances)
                    badX, badY = x, y

    # badCount = max(distances)
    canva[badY, badX] = 2
    if expl:
        print("bad", badX, badY, badCount)

    if ret_images:
        return badCount, [contourDrawed, canva, diff]
    else:
        return badCount


def check_angle_old(pattern, crystal, angle, ret_images=False):
    canva = (
        paste_centrally(np.zeros((400, 400), dtype=np.uint8), pattern) / 255
    ).astype(int)

    # rotating contour
    cntNorm = normalize_crystal(crystal.copy())
    rotated = rotate_contour(cntNorm, angle)

    cntDraw = rotated + 200  # centerize
    contourDrawed = cv2.drawContours(
        np.zeros((400, 400)), [cntDraw], -1, (1), -1
    ).astype(int)
    diff = contourDrawed - canva

    unique, counts = np.unique(diff, return_counts=True)
    badCount = 0
    for u, c in zip(unique, counts):
        if u == -1 or u == 1:
            badCount += c

    if ret_images:
        return badCount, contourDrawed, canva
    else:
        return badCount


def crude_expand_edge(pattern, border=10):
    contours, rects, boxes = get_contours(pattern, circ=0, areas=[30, 32_000])
    cnt, rect = contours[0], rects[0]
    drawn = cv2.drawContours(pattern.copy(), [cnt], -1, (255), border).astype(np.uint8)
    contours, rects, boxes = get_contours(drawn, circ=0, areas=[30, 32_000])
    cnt, rect = contours[0], rects[0]
    return cnt, rect


def check_coords(cX, cY, rad, drawnContour):
    test = cv2.circle(np.zeros_like(drawnContour), (cX, cY), rad, (1), -1)
    if (drawnContour + test).max() == 2:
        return True
    else:
        return False


def fit_angle(pattern, crystal):
    vals = []
    angles = range(0, 360, 5)
    for angle in angles:
        vals.append(check_angle(pattern, crystal, angle))
    return angles[np.argmin(vals)]


def get_illumination_coordinates(crystal, pattern, angle, border=10, ret_images=False):
    canva = (
        paste_centrally(np.zeros((400, 400), dtype=np.uint8), pattern) / 255
    ).astype(np.uint8)
    contours, rects, boxes = get_contours(
        canva, circ=0, areas=[30, 32_000]
    )  # find one contour (the pattern)
    cntPattern, rectPattern = (
        contours[0],
        rects[0],
    )  # take the only one contoru - pattern

    cntNorm = normalize_crystal(crystal)
    cntRotated = rotate_contour(cntNorm, angle)
    crystalImg = cv2.drawContours(
        np.zeros_like(canva), [cntRotated + 200], -1, (1), -1
    ).astype(int)

    expanded, rectExp = crude_expand_edge(canva, border=border)
    drawn = cv2.drawContours(np.zeros_like(canva), [expanded], -1, (1), 1).astype(int)
    coords = np.where(drawn == 1)

    if ret_images:
        return coords, [drawn + canva + crystalImg]
    else:
        return coords


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def get_target(rot: float, n: float, crystalBoundingBox):
    (x, y), (w, h), alfa = crystalBoundingBox
    d = np.sqrt(w**2+h**2)
    deltaX, deltaY = pol2cart(n*0.5*d, np.pi/180*alfa+rot)
    return (int(x+deltaX), int(y+deltaY))
    
    
def get_aoi(img: np.array, aoi: list, size = 800):
    '''
    get area of interest in an image
    img - image
    aoi - [x, y] coords of center
    '''
    x, y = aoi[0], aoi[1]
    half = int(size/2)
    if x-half < 0: x -= x-half
    if x+half >= img.shape[1]: x -= x+half-img.shape[1]
        
    if y-half < 0: y -= y-half
    if y+half >= img.shape[0]: y -= y+half-img.shape[0]
        
    return img[y-half:y+half, x-half:x+half]