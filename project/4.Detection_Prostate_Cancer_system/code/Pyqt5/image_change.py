#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# All imports
import time

import matplotlib.pyplot as plt
import numpy as np
# import openslide
import pandas as pd
import skimage.io
from skimage import morphology
import pprint
import cv2
import json


def otsu_filter(channel, gaussian_blur=True):

    """Otsu filter."""

    if gaussian_blur:
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
    channel = channel.reshape((channel.shape[0], channel.shape[1]))

    return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def detect_tissue(input_slide, sensitivity=3000):

    """
    Description
    ----------
    Find RoIs containing tissue in WSI.
    Generate mask locating tissue in an WSI. Inspired by method used by
    Wang et al. [1]_.
    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew
    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",
    arXiv:1606.05718
    Credit: Github-wsipre

    Parameters
    ----------
    input_slide: numpy array
        Slide to detect tissue on.
    sensitivity: int
        The desired sensitivty of the model to detect tissue. The baseline is set
        at 3000 and should be adjusted down to capture more potential issue and
        adjusted up to be more agressive with trimming the slide.

    Returns (3)
    -------
    -Tissue binary mask as numpy 2D array,
    -Tiers investigated,
    -Time Stamps from running tissue detection pipeline
    """

    # For timing
    time_stamps = {}
    time_stamps["start"] = time.time()

    # Convert from RGB to HSV color space
    slide_hsv = cv2.cvtColor(input_slide, cv2.COLOR_BGR2HSV)
    time_stamps["re-color"] = time.time()
    # Compute optimal threshold values in each channel using Otsu algorithm
    _, saturation, _ = np.split(slide_hsv, 3, axis=2)

    mask = otsu_filter(saturation, gaussian_blur=True)
    time_stamps["filter"] = time.time()
    # Make mask boolean
    mask = mask != 0

    mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)
    mask = morphology.remove_small_objects(mask, min_size=sensitivity)
    time_stamps["morph"] = time.time()
    mask = mask.astype(np.uint8)
    mask_contours, tier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    time_stamps["contour"] = time.time()
    time_stamps = {
        key: (value - time_stamps["start"]) * 1000 for key, value in time_stamps.items()
    }
    return mask_contours, tier, time_stamps


def draw_tissue_polygons(input_slide, tissue_contours, plot_type, line_thickness=None):

    """
    Description
    ----------
    Plot Tissue Contours as numpy array on.
    Credit: Github-wsipre

    Parameters
    ----------
    input_slide: numpy array
        Slide to draw contours onto
    tissue_contours: numpy array
        These are the identified tissue regions as cv2 contours
    plot_type: str ("line" | "area")
        The desired display type for the tissue regions
    line_thickness: int
        If the polygon_type=="line" then this parameter sets thickness

    Returns (1)
    -------
    - Numpy array of tissue mask plotted
    """

    tissue_color = 1

    for cnt in tissue_contours:
        if plot_type == "line":
            output_slide = cv2.polylines(input_slide, [cnt], True, tissue_color, line_thickness)
        elif plot_type == "area":
            if line_thickness is not None:
                warnings.warn(
                    '"line_thickness" is only used if ' + '"polygon_type" is "line".'
                )

            output_slide = cv2.fillPoly(input_slide, [cnt], tissue_color)
        else:
            raise ValueError('Accepted "polygon_type" values are "line" or "area".')

    return output_slide


def tissue_cutout(input_slide, tissue_contours):

    """
    Description
    ----------
    Set all parts of the in_slide to black except for those
    within the provided tissue contours
    Credit: https://stackoverflow.com/a/28759496

    Parameters
    ----------
    input_slide: numpy array
            Slide to cut non-tissue backgound out
    tissue_contours: numpy array
            These are the identified tissue regions as cv2 contours

    Returns (1)
    -------
    - Numpy array of slide with non-tissue set to black
    """

    # Get intermediate slide
    base_slide_mask = np.zeros(input_slide.shape[:2])

    # Create mask where white is what we want, black otherwise
    crop_mask = np.zeros_like(base_slide_mask)

    # Draw filled contour in mask
    cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1)

    # Extract out the object and place into output image
    tissue_only_slide = np.zeros_like(input_slide)
    tissue_only_slide[crop_mask == 255] = input_slide[crop_mask == 255]

    return tissue_only_slide


def getSubImage(input_slide, rect):

    """
    Description
    ----------
    Take a cv2 rectagle object and remove its contents from
    a source image.
    Credit: https://stackoverflow.com/a/48553593

    Parameters
    ----------
    input_slide: numpy array
            Slide to pull subimage off
    rect: cv2 rect
        cv2 rectagle object with a shape of-
            ((center_x,center_y), (hight,width), angle)

    Returns (1)
    -------
    - Numpy array of rectalge data cut from input slide
    """

    width = int(rect[1][0])
    height = int(rect[1][1])
    box = cv2.boxPoints(rect)

    src_pts = box.astype("float32")
    dst_pts = np.array(
        [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    output_slide = cv2.warpPerspective(input_slide, M, (width, height))
    return output_slide


def color_cut(in_slide, color = [255,255,255]):

    """
    Description
    ----------
    Take a input image and remove all rows or columns that
    are only made of the input color [R,G,B]. The default color
    to cut from image is white.

    Parameters
    ----------
    input_slide: numpy array
        Slide to cut white cols/rows
    color: list
        List of [R,G,B] pixels to cut from the input slide

    Returns (1)
    -------
    - Numpy array of input_slide with white removed
    """
    #Remove by row
    row_not_blank = [row.all() for row in ~np.all(in_slide == color, axis=1)]
    output_slide = in_slide[row_not_blank, :]

    #Remove by col
    col_not_blank = [col.all() for col in ~np.all(output_slide == color, axis=0)]
    output_slide = output_slide[:, col_not_blank]
    return output_slide

def detect_and_crop(image_location,sensitivity= 3000,downsample_lvl= -1,show_plots= "simple"):

    """
    Description
    ----------
    This method performs the pipeline as described in the notebook:
    https://www.kaggle.com/dannellyz/panda-tissue-detection-size-optimization-70

    Parameters
    ----------
    image_location:str
        Location of the slide image to process
    sensitivity:int
        The desired sensitivty of the model to detect tissue. The baseline is set
        at 3000 and should be adjusted down to capture more potential issue and
        adjusted up to be more agressive with trimming the slide.
    downsample_lvl: int
        The level at which to downsample the slide. This can be referenced in
        reverse order to access the lowest resoltuion items first.
        [-1] = lowest resolution
        [0] = highest resolution
    show_plots: str (verbose|simple|none)
        The types of plots to display:
            - verbose - show all steps of process
            - simple - show only last step
            - none - show none of the plots

    Returns (4)
    -------
    - Numpy array of final produciton(prod) slide
    - Percent memory reduciton from original slide
    - Time stamps from stages of the pipeline
    - Time stamps from the Tissue Detect pipeline
    """

    # For timing
    time_stamps = {}
    time_stamps["start"] = time.time()

    # Open Slide
    wsi = skimage.io.MultiImage(image_location)[downsample_lvl]
    time_stamps["open"] = time.time()

    # Get returns from detect_tissue()
    (
        tissue_contours,
        tier,
        time_stamps_detect,
    ) = detect_tissue(wsi, sensitivity)
    time_stamps["tissue_detect"] = time.time()
    # Get Tissue Only Slide
    base_slide_mask = np.zeros(wsi.shape[:2])
    tissue_slide = draw_tissue_polygons(base_slide_mask, tissue_contours, "line", 5)
    tissue_only_slide = tissue_cutout(wsi, tissue_contours)
    time_stamps["tissue_trim"] = time.time()
    # Get minimal bounding rectangle for all tissue contours
    if len(tissue_contours) == 0:
        img_id = image_location.split("/")[-1]
        print(f"No Tissue Contours - ID: {img_id}")
        return None, 1.0

    all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))
    # Crop with getSubImage()
    smart_bounding_crop = getSubImage(tissue_only_slide, all_bounding_rect)
    time_stamps["crop"] = time.time()

    #cut empty space
    prod_slide = color_cut(smart_bounding_crop, [0,0,0])
    time_stamps["trim_white"] = time.time()

    # Get size change
    base_size = get_disk_size(wsi)
    final_size = get_disk_size(prod_slide)
    pct_change = final_size / base_size
    if show_plots == "simple":
        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")
        plt.imshow(smart_bounding_crop)
        plt.show()
    elif show_plots == "verbose":
        # Set-up dictionary for plotting
        verbose_plots = {}
        # Add Base Slide to verbose print
        verbose_plots[f"Base Slide\n{get_disk_size(wsi):.2f}MB"] = wsi
        # Add Tissue Only to verbose print
        verbose_plots[f"Tissue Detect\nNo Change"] = tissue_slide
        # Add Bounding Boxes to verbose print
        verbose_plots[
            f"Bounding Boxes\n{get_disk_size(smart_bounding_crop):.2f}MB"
        ] = smart_bounding_crop
        # Add Cut Slide to verbose print
        verbose_plots[
            f"Cut Slide\n{get_disk_size(prod_slide):.2f}MB"
        ] = prod_slide
        print(f"Percent Reduced from Base Slide to Final: {(1- pct_change)*100:.2f}")
        plot_figures(verbose_plots, 2, 2)
    elif show_plots == "none":
        pass
    else:
        pass
    time_stamps = {
        key: (value - time_stamps["start"]) * 1000 for key, value in time_stamps.items()
    }
    return prod_slide, (1 - pct_change), time_stamps, time_stamps_detect


def get_disk_size(numpy_image):
    """Return disk size of a numpy array"""
    return (numpy_image.size * numpy_image.itemsize) / 1000000


def plot_figures(figures, nrows=1, ncols=1):

    """
    Description
    ----------
    Plot a dictionary of figures.
    Credit: https://stackoverflow.com/a/11172032

    Parameters
    ----------
    figures: dict
        <title, figure> for those to plot
    ncols: int
        number of columns of subplots wanted in the display
    nrows: int
        number of rows of subplots wanted in the figure

    Returns(0)
    ----------
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], aspect="auto")
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout()
    plt.show()
    return


def get_timings(time_stamp_dict, verbose = False):
    """
    Description
    ----------
    Get timing defferentials and percentages from a dictionary of timestamps

    Parameters
    ----------
    time_stamp_dict: dict
        <description:time> for each of the measured time points

    Returns (2)
    -------
    - Dictionary with the time differentials at each time point
    - Dictionary with the time percentages for each point compared to total
    """

    time_diffs = {}
    dict_list = list(time_stamp_dict.items())
    for i in range(len(dict_list) - 1):
        time_diffs[dict_list[i + 1][0]] = dict_list[i + 1][1] - dict_list[i][1]
    total_time = list(time_stamp_dict.values())[-1]
    time_pcts = {k: v / (total_time + 0.0001) for k, v in time_diffs.items()}
    if verbose:
        print(f"Total Between Funcitons:")
        print(json.dumps(time_diffs, indent=4))
        print(f"Pct Between Funcitons:")
        print(json.dumps(time_pcts, indent=4))
        print(f"Timing Totals:")
        print(json.dumps(time_stamp_dict, indent=4))
    return time_diffs, time_pcts

def comp_timings(time_stamp_low, time_stamp_high, verbose=False):
    """
    Description
    ----------
    Take two timestamp dictionaries and compare them.

    Parameters
    ----------
    time_stamp_low: dict
        <description:time> for each of the measured time points (shorter process)
    time_stamp_high: dict
        <description:time> for each of the measured time points (longer process)

    Returns (1)
    -------
    - Dictionary with the raw differences between the two dicts
    - Dictionary with the percentage differences between the two dicts
    """

    raw_comp = {k:time_stamp_high[k]-time_stamp_low[k] for k,v in time_stamp_low.items()}
    pct_comp = {k:time_stamp_low[k]/time_stamp_high[k] for k,v in time_stamp_low.items()}
    if verbose:
        print(f"Timing Diffs Raw (High - Low):")
        print(json.dumps(raw_comp, indent=4))
        print(f"Timing: Diffs Pct (Low / High):")
        print(json.dumps(pct_comp, indent=4))
    return raw_comp, pct_comp





file_path = 'C:/Users/user/Desktop/QT/image/test1_1.png'

def image_change_view(file_path):
    # Open slide on lowest resolution
    low_res_lvl = 0
    img_low = skimage.io.MultiImage(file_path)[low_res_lvl]

    # ## Detect Tissue and Find Bounding Boxes

    # Detect Tissue and
    (
        tissue_contours,
        tier,
        time_stamps_detect,
    ) = detect_tissue(img_low)
    # Copy for compare
    smart_bounding_boxes = img_low.copy()
    # Get small level bounding boxes
    all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_contours))
    all_bounding_box = cv2.boxPoints(all_bounding_rect)
    all_bounding_box = np.int0(all_bounding_box)


    # ## Scale Smaller Bounding Boxes to High Res Image


    smart_scale = img_low.copy()
    scale = 1
    # Get center, size, and angle from rect
    scaled_rect = (
        (all_bounding_rect[0][0] * scale, all_bounding_rect[0][1] * scale),
        (all_bounding_rect[1][0] * scale, all_bounding_rect[1][1] * scale),
        all_bounding_rect[2],
    )
    scaled_bounding_box = cv2.boxPoints(scaled_rect)
    scaled_bounding_box = np.int0(scaled_bounding_box)
    cv2.drawContours(smart_scale, [scaled_bounding_box], 0, (255, 255, 255), 0)
    # ## Crop High Res Image with Bounding Box

    scaled_smart_crop = getSubImage(smart_scale, scaled_rect)

    # ## Crop White Space From Image
    big_slide_cut = color_cut(scaled_smart_crop, color=[255,255,255])

    print(type(big_slide_cut))
    # plt.imshow(big_slide_cut)
    # plt.show()
    # image = np.array(big_slide_cut, dtype=np.float32).tobytes()
    # print(type(image))
    cv2.imwrite("output.png",big_slide_cut)


