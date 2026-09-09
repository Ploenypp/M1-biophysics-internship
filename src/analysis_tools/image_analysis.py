import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2

# load images for poisson ratio 
def load_images_poisson_ratio(path) :
    nb_frames = len(os.listdir(path))-1
    frame_limit = int(nb_frames*(3/4)) + 1
    step = 100
    if frame_limit/step < 10 : step = 10
    frame_names = [os.listdir(path)[i] for i in range(0, frame_limit, step)]

    img_lst = []
    for frame in frame_names :
        img_path = Path(path) / frame
        img_lst.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
    return img_lst

# load images for high strain analysis
def load_images_high_strain(path, step=True) :
    nb_frames = len(os.listdir(path))-1

    steps = 1
    if step :
        if nb_frames/10 > 100 : steps = 10
    
    frame_names = sorted([os.listdir(path)[i] for i in range(0, nb_frames, steps)])

    frames = []
    for i in range(len(frame_names)-1) :
        f = frame_names[i]
        if f.split(".")[1] == "tif" :
            print(f"\rloading images {i+1}/{len(frame_names)-1}", end="", flush=True)
            frame_path = Path(path) / f
            frames.append(cv2.imread(frame_path, cv2.IMREAD_UNCHANGED))
    print()
    return frames

# find rod borders for cropping and lateral strain measurements
def get_rod_borders(img) :
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, borders = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find candidate indexes for for borders
    midX = int(img.shape[1]/2)
    oL_candidates = []; iL_candidates = []; iR_candidates = []

    # find rod border x-coordinate candidates
    for i in range(img.shape[0]) :
        row = borders[i,:]
        prev_leftmost = row[0]; prev_mid = row[midX]

        # find outer left candidates
        for j in range(1, img.shape[0]) :
            if row[j] != prev_leftmost : # only take the first different value
                oL_candidates.append(j)
                break

        # find inner left candidates
        for j in range(midX, 1, -1) :
            if row[j] != prev_mid : # only take the first different value
                iL_candidates.append(j)
                break
        
        # find inner right candidates
        for j in range(midX, img.shape[1]) :
            if row[j] != prev_mid : # only take first different value
                iR_candidates.append(j)
                break
    
    oL = int(np.median(sorted(oL_candidates)))

    iL_candidates = dict(sorted(Counter(sorted(iL_candidates)).items(), key=lambda item: item[1], reverse=True))
    iR_candidates = dict(sorted(Counter(sorted(iR_candidates)).items(), key=lambda item: item[1], reverse=True))

    iL = list(iL_candidates.keys())[0]
    iR = list(iR_candidates.keys())[0]

    nw = list(iL_candidates.items())[0][1]; sw = list(iL_candidates.items())[-1][1]
    ne = list(iR_candidates.items())[0][1]; se = list(iR_candidates.items())[-1][1]

    left_slope = (nw - sw)/img.shape[0]
    right_slope = (ne - se)/img.shape[0]

    left_border = [(x, int(x*left_slope)) for x in range(img.shape[1])]
    right_border = [(x, int(x*right_slope)) for x in range(img.shape[1])]

    return oL, iL, iR, left_border, right_border

# returns blurred image for subsequence rod border detection and binarized image
def process_image(img, check=False) :
    # depending on which microscope the image was taken, the parameters and process change

    # prepare and binarize images
    
    # smooth by blurring to reduce noise
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # highlight dark features on light background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    # detect only horizontal edges
    horizontal_edges = cv2.Sobel(blackhat, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(np.zeros(img.shape), horizontal_edges)
    sobel = cv2.convertScaleAbs(sobel)

    # binarize
    _, binarized = cv2.threshold(sobel, 250, 255, cv2.THRESH_BINARY)

    # check edges
    if check : plt.imshow(binarized, cmap="gray")

    # get rod borders and crop
    oL, iL, iR, left_border, right_border = get_rod_borders(blur)
    cropped = binarized[0:img.shape[0], iL:iR]

    return cropped, (oL, iL, iR), (left_border, right_border)

