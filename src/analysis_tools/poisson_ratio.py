from math import dist

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.morphology import skeletonize

# get skeleton coordinates
def get_gel_borders(img, check=False) :
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    skeleton = []

    dilated = cv2.dilate(img, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.RETR_TREE retrieves all contours, including nested
        # cv2.CHAIN_APPROX_SIMPLE removes redundant points

    if check :
        contoured = img.copy()
        contoured = cv2.cvtColor(contoured, cv2.COLOR_GRAY2BGR)
        plt.imshow(cv2.drawContours(contoured, contours, -1, (5, 200, 250), 2))
        plt.show()

    # calculate areas from contours
    areas = [cv2.contourArea(x) for x in contours]

    # get contour(s) with the largest area(s)
    border_contours = []
    max_area1 = -1; max_area2 = -1
    for i in range(len(contours)) :
        if areas[i] > max_area1 :
            if max_area1 > max_area2 :
                max_area2 = max_area1
                contour2 = contour1

            max_area1 = areas[i]
            contour1 = contours[i]

        elif areas[i] > max_area2 :
            max_area2 = areas[i]
            contour2 = contours[i]
    border_contours = [contour1, contour2] 

    if check :
        contoured = img.copy()
        contoured = cv2.cvtColor(contoured, cv2.COLOR_GRAY2BGR)
        plt.imshow(cv2.drawContours(contoured, border_contours, -1, (5, 200, 250), 2))
        plt.show()

    # draw and fille contours, then skeletonize
    skeleton = np.zeros(img.shape) # cropped.copy()
    cv2.fillPoly(skeleton, border_contours, (255,255,255))

    skeleton = skeletonize(skeleton)

    return sorted([(x,y) for y in range(skeleton.shape[0]) for x in range(skeleton.shape[1]) if skeleton[y,x]])

# reduces curve to single layer (take the median y-value for a single x)
def single_layer_curve(curve) :
    """
    Some x-values may still have more than 1 y-value after skeletonizing. In such cases, it was decided to take the median of those values.
    """

    xy_dict = dict()
    for (x,y) in curve :
        if x not in xy_dict : xy_dict[x] = [y]
        else : xy_dict[x].append(y)
    
    return [(x, int(np.median(y))) for x,y in xy_dict.items()]

# returns curve coordinates
def get_curves(coordinates) :
    x,y = zip(*coordinates) 

    # calculate midpoint y to separate curves
    y_threshold = int(np.median([min(y), max(y)]))

    # render coordinates into single layer curves and split into x and y
    upper_x, upper_y = zip(*single_layer_curve([coordinates[i] for i in range(len(coordinates)) if y[i] <= y_threshold]))
    lower_x, lower_y = zip(*single_layer_curve([coordinates[i] for i in range(len(coordinates)) if y[i] > y_threshold]))
    
    return upper_x, upper_y, lower_x, lower_y

# aligns x-coordinates of 2 curves (upper and lower)
def get_aligned_upper_lower_idx(upper_x, lower_x) :
    """
    The lengths of the upper and lower curves are not equivalent thus the first element in each list may not correspond to the same x-value. For methods where both curves are scanned simultaneously, determining the frame of indexes where they both have values and aligning their indexes is necessary.
    """

    # find the lower and upper bounds for x
    start = max([min(upper_x), min(lower_x)])
    stop = min([max(upper_x), max(lower_x)])

    upper_start = -1; lower_start =-1

    # find the index where the upper curve starts
    for i in range(len(upper_x)) :
        if upper_x[i] == start :
            upper_start = i
            break

    # find the index where the lower curve starts
    for i in range(len(lower_x)) :
        if lower_x[i] == start :
            lower_start = i
            break

    # calculate the stop indexes
    upper_stop = upper_start + abs(stop - start)
    lower_stop = lower_start + abs(stop - start)

    # generate the list of indexes in frame for both curves
    upper_idx = np.arange(upper_start, upper_stop, 1)
    lower_idx = np.arange(lower_start, lower_stop, 1)

    # pair upper and lower indexes
    return list(zip(upper_idx, lower_idx))

# === width calculation methods for poisson ratio calculations === #
"""
Each method returns a tuple (extrema, distance1, distance2)

extrema is the coordinates of the minima and maxima of the upper and lower curves, respectively.

Typically, distance1 is the absolute difference between the y-coordinates of the extrema and distance2 is the calculated distance between the extrema.
"""

def get_min_width(upper_x, upper_y, lower_x, lower_y) :
    """
    This method scans both curves simultaneously within a frame where they both have values to find the minimum width. The minimum width is the difference between the y-values.
    """

    aligned_idx = get_aligned_upper_lower_idx(upper_x, lower_x)

    # calculate widths
    widths = {(a,b):abs(upper_y[a] - lower_y[b]) for a,b in aligned_idx}
    min_width = min(widths.values()) # find minimum width

    # find indexes corresponding to minimum width
    upper_idx, lower_idx = -1, -1
    for c,w in widths.items() :  
        if w == min_width :
            upper_idx, lower_idx = c
            break
    
    extrema = ((upper_x[upper_idx], upper_y[upper_idx]), (lower_x[lower_idx], lower_y[lower_idx]))

    return extrema, min_width, dist(extrema[0], extrema[1])

def get_min_max_y_diff(upper_x, upper_y, lower_x, lower_y) :
    """
    This method finds the upper curve's minimum y-value and the lower curve's maximum y-value and takes the difference. To determine extrema' x-values, the median of the x-values corresponding to the associated y-values is used.
    """

    muy = max(upper_y); mly = min(lower_y)
    
    # find median x-values for muy and mly
    mux = [upper_x[i] for i in range(len(upper_y)) if upper_y[i] == muy]
    mlx = [lower_x[i] for i in range(len(lower_y)) if lower_y[i] == mly]

    extrema = ((int(np.median(mux)), muy), (int(np.median(mlx)), mly))

    return extrema, abs(muy - mly), dist(extrema[0], extrema[1])

def get_median_x(upper_x, upper_y, lower_x, lower_y) :
    """
    This method uses the median x-value of each curve to find the y-values to be used to calculate the width. In the case there are multiple y-values (which there shouldn't be), the median is taken.
    """

    mux = int(np.median(upper_x)); mlx = int(np.median(lower_x))
    
    # find median y-value for mux and mlx
    muy = [upper_y[i] for i in range(len(upper_x)) if upper_x[i] == mux]
    mly = [lower_y[i] for i in range(len(lower_x)) if lower_x[i] == mlx]

    extrema = ((mux, int(np.median(muy))), (mlx, int(np.median(mly))))

    return extrema, abs(extrema[0][1] - extrema[1][1]), dist(extrema[0], extrema[1])

def get_median_of_median_x(upper_x, upper_y, lower_x, lower_y) :
    """
    This method is like get_median_x() except it uses the median of the curves' median x-value.
    """

    mux = int(np.median(upper_x)); mlx = int(np.median(lower_x))
    mx = int(np.median([mux, mlx]))

    # find median y-value for mx
    muy = [upper_y[i] for i in range(len(upper_x)) if upper_x[i] == mx]
    mly = [lower_y[i] for i in range(len(lower_x)) if lower_x[i] == mx]

    if len(muy) == 0 :
        muy = [upper_y[i] for i in range(len(upper_x)) if upper_x[i] in range(mx-5, mx+5)]
    
    if len(mly) == 0 :
        mly = [lower_y[i] for i in range(len(lower_x)) if lower_x[i] in range(mx-5, mx+5)]

    extrema = (mx, int(np.median(muy))), (mx, int(np.median(mly)))

    return extrema, abs(extrema[0][1] - extrema[1][1]), dist(extrema[0], extrema[1])

def get_mean_width(upper_x, upper_y, lower_x, lower_y) :
    aligned_idx = get_aligned_upper_lower_idx(upper_x, lower_x)

    # calculate widths
    widths = {(a,b):abs(upper_y[a] - lower_y[b]) for a,b in aligned_idx}
    mean_width = np.mean(list(widths.values())) # find mean width

    # find indexes corresponding to mean width
    upper_idx, lower_idx = -1, -1
    for c,w in widths.items() :  
        if w == mean_width :
            upper_idx, lower_idx = c
            break

    # 4 points to make a box 
    # upper left, upper right, lower left, lower right
    nw = (upper_x[aligned_idx[0][0]], upper_y[upper_idx])
    ne = (upper_x[aligned_idx[-1][0]]), upper_y[upper_idx]

    sw = (lower_x[aligned_idx[0][1]], lower_y[lower_idx])
    se = (lower_x[aligned_idx[-1][1]], lower_y[lower_idx])

    extrema = [nw, ne, sw, se]
    
    return extrema, round(float(mean_width),3), round(float(mean_width),3)

# === end of width calculation methods === #

# calculate strain
def calculate_strain(lengths) : 
    fst = lengths[0]

    if fst == 0 : fst = 0.00001
    return [(l-fst)/fst for l in lengths[1:]]

# calculate poisson ratio from transverse and lateral strain
def calculate_poisson_ratio(lateral, transverse, limit=0.4) :
    lateral_strain = calculate_strain(lateral)
    transverse_strain = calculate_strain(transverse)

    i = 0
    # for poisson ratio, we only calculated up to 40% strain
    while i < len(lateral_strain) and lateral_strain[i] <= 0.4 :
        i += 1

    poisson, y_int = np.polyfit(lateral_strain[:i], transverse_strain[:i], deg=1)

    return -poisson

# mark peaks 
def mark_width(img, peaks_lst) :
    marked = img.copy()
    cv2.cvtColor(marked, cv2.COLOR_GRAY2BGR)

    for (x,y) in peaks_lst :
        # draw each "peak"
        cv2.drawMarker(
            marked, 
            (x,y), 
            (255,200,0), 
            markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2
        )
    # draw a line between "peaks"
    cv2.line(
        marked,
        peaks_lst[0], peaks_lst[1],
        (255,200,0), thickness=1
    )

    # in the case that the mean width was taken
    if len(peaks_lst) > 2 :
        nw = peaks_lst[0]; ne = peaks_lst[1]
        sw = peaks_lst[2]; se = peaks_lst[3]

        cv2.polylines(
            marked,
            [nw, ne, sw, se],
            (255,200,0), thickness=1
        )

    return marked

# check chosen widths
def check_widths(img_lst, img_peaks) :
    widths = [img.shape[1] for img in img_lst]

    fig, axs = plt.subplots(1, len(img_lst), figsize=(5,5), sharey=True, gridspec_kw={"width_ratios" : widths})

    for i in range(len(img_lst)) :
        axs[i].imshow(img_lst[i], cmap="gray", aspect="equal")
        axs[i].set_title(f"{i * 10}%")
        if i > 0 : axs[i].yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

# check poisson ratio and deformations
def check_poisson(lateral, transverse, poisson, y_int) :
    lateral_strain = calculate_strain(lateral)
    transverse_strain = calculate_strain(transverse)

    fig = plt.figure(figsize=(3,3))
    
    # strains
    plt.scatter(lateral_strain, transverse_strain, color="blue")
    plt.plot(lateral_strain, transverse_strain, color="blue")

    # check poisson ratio
    plt.plot(
        lateral_strain, 
        np.multiply(lateral_strain, poisson) + y_int, 
        color="red", linestyle="--", linewidth=1
    )

    plt.xlabel("lateral strain")
    plt.ylabel("transverse strain")
    plt.title(f"poisson ratio = {round(poisson,3)}")

    return fig

