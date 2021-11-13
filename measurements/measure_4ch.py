
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from skimage.transform import hough_line
from scipy.ndimage.measurements import center_of_mass
from itertools import product
from scipy.ndimage.morphology import binary_fill_holes
import multiprocessing
from skimage.color import gray2rgb
from skimage import io
from functools import partial

from measurements.image import mask_data
from measurements.zip import load_zip_data, load_zip_pixelspacing
from utils.filepath import secure_folder


##################################################
# Utilities for plotting and saving figs
##################################################
def save_masks(rgb, sep_mask, figname):
    rgb_masked = mask_data(rgb, sep_mask)
    io.imsave(figname, rgb_masked)

def scatter_pts(img, pts, figsize=(12,12)):
    """
    The utility to scatter pts on the image
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    for pt in pts:
        plt.scatter(pt[1], pt[0], s=100, marker='X')
    plt.show()

def draw_arrows(mask, arrow_pts, arrow_width=3, arrow_length=3, colors=['pink'], figname=None):
    """
    Utility to draw arrows on the mask
    """
    def get_arrow_pts(pts):
        mid_pt = np.mean(pts, axis=0)
        diff = pts[0] - mid_pt
        diff_length = np.linalg.norm(diff,2)
        diff = diff * (diff_length-arrow_length)/diff_length
        return mid_pt, diff
    
    if len(colors) < len(arrow_pts):
        colors = colors + [colors[np.random.randint(len(colors))] for i in range(len(arrow_pts)-len(colors))]
    
    plt.figure(figsize=(12,12))
    plt.imshow(mask)
    for i, arrow_pt in enumerate(arrow_pts):
        mid_pt, diff = get_arrow_pts(arrow_pt)
        color = colors[i]
        plt.arrow(mid_pt[1], mid_pt[0], diff[1], diff[0], fc=color, ec=color,
                  width=arrow_width/3, head_width=arrow_width, head_length=arrow_length)
        plt.arrow(mid_pt[1], mid_pt[0], -diff[1], -diff[0], fc=color, ec=color,
                  width=arrow_width/3, head_width=arrow_width, head_length=arrow_length)

    if figname is not None:
        plt.savefig(figname)
    else:
        plt.show()

##################################################
# Getters for mask properties
##################################################
def get_pts_distance(pt1, pt2):
    """
    Get euclidean distance between two points
    """
    return np.linalg.norm(pt1-pt2, ord=2)

def get_pts_distance_with_pix(pts, pix):
    """
    Get euclidean distance of two points with pixelspacing
    """
    return np.linalg.norm((pts[0]-pts[1])*pix, ord=2)

def get_pt_ct_distance(pt, ct):
    """
    Get the distance between a point [pt] and a contour [ct]

    Params
    ------
    pt      : input point [y, x]
    ct      : input contour, an array of points [xx, 2]

    Return
    ------
    minimum_distance                : float
    closest_point_index             : int
    closest_point_on_the_contour    : (int, int)
    input_point                     : (int, int)
    """
    distance = np.array([get_pts_distance(pt, pt_i) for pt_i in ct])
    ct_i = np.argmin(distance)
    return np.array([distance[ct_i], ct_i, *ct[ct_i], *pt])


def get_line_end_pts(line):
    """
    Get the two furthest points from the middle point of the line
    """
    mid_pt = np.mean(line, axis=0)
    
    distances = np.array([get_pts_distance(mid_pt, pt_x) for pt_x in line])
    pt_i = np.argmax(distances)
    distances = np.array([get_pts_distance(line[pt_i], pt_x) for pt_x in line])
    pt_j = np.argmax(distances)
    
    end_pts = line[np.array([pt_i, pt_j])]
    
    return end_pts


def connect_pts(pt1, pt2):
    """
    Connect two points and return the connecting line
    starting with pt1, ending with pt2
    """
    steps = int(np.sum(np.abs(pt2 - pt1)))
    y_step = 1 if pt2[0] > pt1[0] else -1
    x_step = 1 if pt2[1] > pt1[1] else -1
    pts = [pt1]
    
    for step in range(steps-1):
        delta = np.abs(pt2 - pts[-1])
        y_ratio = delta[0] / delta.sum()
        pt_new = np.copy(pts[-1])
        if np.random.rand() <= y_ratio:
            pt_new[0] = pt_new[0] + y_step
        else:
            pt_new[1] = pt_new[1] + x_step
        pts.append(pt_new)
    pts.append(pt2)
    
    return np.array(pts)

def get_thetas(mask):
    """
    Get thetas of lines connecting 4 chambers 
        - theta of line RA -> LA
        - theta of line RV -> RA
        - theta of line LV -> RV
        - theta of line LA -> LV
    """
    cs = [center_of_mass(mask==i) for i in [1,2,3,4,1]] # (y,x)
    
    thetas = []
    for i in range(4):
        theta = np.arctan2(cs[i][0]-cs[i+1][0], cs[i][1]-cs[i+1][1])
        thetas.append(theta)
        
    return thetas
    
def get_labels(mask, background=0):
    """
    Utility for getting labels of the mask
        - background is labeled 0.
    """
    labels = np.unique(mask)
    labels = labels[labels!=background]
    return labels
        
    
def get_contours(mask):
    """
    Get the contour of the labeled mask

    Params
    ------
    mask    : input mask, np.unique(mask) = [0, 1, 2, ...]
            : background pixels are labeled 0.

    Return
    ------
    contours: contours corresponding to the labels in the mask.
    """

    contours = []
    for label in get_labels(mask):
        contour = measure.find_contours(mask==label, level=0)[0]
        contours.append(contour)
    return contours

def get_centers(mask):
    """
    Get the center of mass of the labeled areas in the mask
    """
    centers = np.array([center_of_mass(mask==label) for label in get_labels(mask)])
    return centers

def find_end_pts(mask, theta, plot=False, anchor=None):
    """
    Find the end points of the mask along the direction of theta
        - when anchor is given, return the end point further from anchor

    Params
    ------
    mask    : the input contour [xx, 2] or mask [width, height].
    theta   : the direction of the centerline in radian
    plot    : trigger to plot the results
    anchor  : the anchor point to get the further end point

    Return
    ------
    end_points

    """
    if mask.shape[1] == 2:
        contour = np.copy(mask).astype(int)
        mask = np.zeros((256,256))
        mask[contour[:, 0], contour[:, 1]] = 1
        
    yy, xx = np.where(mask>0)
    rr = xx * np.cos(theta) + yy * np.sin(theta)
    
    r0 = np.argmin(rr)
    r1 = np.argmax(rr)
    
    pt0 = np.array([yy[r0], xx[r0]])
    pt1 = np.array([yy[r1], xx[r1]])
    
    if plot:
        scatter_pts(mask, [pt0, pt1])

    if anchor is not None:
        if get_pts_distance(pt0, anchor) > get_pts_distance(pt1, anchor):
            return pt0
        else:
            return pt1
    else:
        return pt0, pt1

def find_mask_end_pts(mask):
    """
    Find the tip points of the 4 heart chambers
    aka. the points away from the valves.
    in the order [LA, RA, RV, LV]
    """
    thetas = get_thetas(mask)
    contours = get_contours(mask)
    centers = get_centers(mask)
    
    end_pts_1 = find_end_pts(contours[0], thetas[3], anchor=centers[3])
    end_pts_2 = find_end_pts(contours[1], thetas[1], anchor=centers[2])
    end_pts_3 = find_end_pts(contours[2], thetas[1], anchor=centers[1])
    end_pts_4 = find_end_pts(contours[3], thetas[3], anchor=centers[0])
    
    end_pts = np.array([end_pts_1, end_pts_2, end_pts_3, end_pts_4])
    
    return end_pts

    
##################################################
# Various functions for measuring the chambers
##################################################

def find_contour_contact(ct1, ct2, margin=2):
    ctt1 = np.array([get_pt_ct_distance(pt, ct2) for pt in ct1])
    # ctt1: [distance, dest_pt_i, dest_pt_y, dest_pt_x, orig_pt_y, orig_pt_x]
    uniq_links = np.unique(ctt1[:,1])
    ctt1_filtered = []
    for uniq_pt in uniq_links:
        pts = np.where(ctt1[:,1]==uniq_pt)[0]
        ctt1_filtered.append(pts[np.where(ctt1[pts][:,0] < ctt1[pts][:,0].min() + margin)[0]])
        
    ctt1f = np.sort(np.concatenate(ctt1_filtered)).astype(int)
    #ctt1s = ctt1[ctt1f]
    
    ct1 = ct1.astype(int)
    line_img = np.zeros((256,256))
    line_img[ct1[ctt1f][:,0], ct1[ctt1f][:,1]] = 1
    label = measure.label(line_img)
    props = measure.regionprops(label)
    
    line = props[np.argmax([x.area for x in props])].coords
    #print(line)
    pts = get_line_end_pts(line)
    #c2 = np.mean(ct2, axis=0)
    #pts_distance = [get_pts_distance(c2, pt_i) for pt_i in pts]
    #pts = pts[np.argsort(pts_distance)]
    
    return pts, line

def find_contour_contact2(ct1, ct2, valve_pt, sep_cc, margin=2):
    ctt1 = np.array([get_pt_ct_distance(pt, ct2) for pt in ct1])
    # ctt1: [distance, dest_pt_i, dest_pt_y, dest_pt_x, orig_pt_y, orig_pt_x]
    uniq_links = np.unique(ctt1[:,1])
    ctt1_filtered = []
    for uniq_pt in uniq_links:
        pts = np.where(ctt1[:,1]==uniq_pt)[0]
        ctt1_filtered.append(pts[np.where(ctt1[pts][:,0] < ctt1[pts][:,0].min() + margin)[0]])
        
    ctt1f = np.sort(np.concatenate(ctt1_filtered)).astype(int)
    #ctt1s = ct1[ctt1f]
    
    ct1 = ct1.astype(int)
    line_img = np.zeros((256,256))
    line_img[ct1[ctt1f][:,0], ct1[ctt1f][:,1]] = 1
    label = measure.label(line_img)
    props = measure.regionprops(label)
    
    line = props[np.argmax([x.area for x in props])].coords
    peak_pts = get_line_end_pts(line)
    
    # Second part : getting valve point from valve lines.
    if np.ndim(valve_pt) == 2:
        valve_pt_distance = [get_pts_distance(sep_cc, pt_i) for pt_i in valve_pt]
        valve_pt = valve_pt[np.argmin(valve_pt_distance)]
        
    peak_pt_distance = [get_pts_distance(valve_pt, pt_i) for pt_i in peak_pts]
    peak_pt = peak_pts[np.argmax(peak_pt_distance)]
        
    contour_img = np.zeros((256,256))
    contour_img[ct1[:,0], ct1[:,1]] = 1
    contour_img[int(valve_pt[0]), int(valve_pt[1])] = 0
    contour_img[int(peak_pt[0]), int(peak_pt[1])] = 0
    
    label = measure.label(contour_img)
    props = measure.regionprops(label)
    line_distance = [get_pts_distance(sep_cc, x.centroid) for x in props]
    line = props[np.argmin(line_distance)].coords
    pts = np.array([peak_pt, valve_pt])
    
    return pts, line


def clean_mask(mask):
    label = measure.label(mask)
    props = measure.regionprops(label)
    label_id = np.argmax([x.area for x in props]) + 1
    cleaned_mask = (label == label_id).astype(int)
    
    return cleaned_mask

def get_septal_mask(septalL, septalR, mask):
    septalL_pts, septalL_line = septalL
    septalR_pts, septalR_line = septalR
    pairs = []
    for pt1, pt2 in product(septalL_pts, septalR_pts):
        distance = get_pts_distance(pt1, pt2)
        pairs.append([distance, *pt1, *pt2])
    pairs = np.array(pairs)
    pairs = pairs[np.argsort(pairs[:,0])[:2]]
    parts1 = connect_pts(pairs[0, 1:3], pairs[0, 3:5])
    parts2 = connect_pts(pairs[1, 1:3], pairs[1, 3:5])
    mid_pts = np.mean(pairs[:2, 1:5].reshape(2,2,2), axis=1)
    
    septal_contour = np.concatenate([septalL_line, septalR_line, parts1, parts2]).astype(int)
    septal_mask = np.zeros((256,256))
    septal_mask[septal_contour[:,0], septal_contour[:,1]] = 1
    septal_mask = binary_fill_holes(septal_mask)
    septal_mask[np.where(mask>0)] = 0
    
    septal_mask = clean_mask(septal_mask)
    
    return septal_mask, mid_pts

def process_mask1(mask):
    """
    Use close contact to find both peak and valve points
    """
    #print(f"processing {pid} .. ")
    contours = get_contours(mask)
    
    mitral_LA = find_contour_contact(contours[0], contours[3])
    mitral_LV = find_contour_contact(contours[3], contours[0])
    tricus_RA = find_contour_contact(contours[1], contours[2])
    tricus_RV = find_contour_contact(contours[2], contours[1])
    atrsep_LA = find_contour_contact(contours[0], contours[1])
    atrsep_RA = find_contour_contact(contours[1], contours[0])
    vensep_LV = find_contour_contact(contours[3], contours[2])
    vensep_RV = find_contour_contact(contours[2], contours[3])
    
    asep_mask, asep_pts = get_septal_mask(atrsep_LA, atrsep_RA, mask)
    vsep_mask, vsep_pts = get_septal_mask(vensep_LV, vensep_RV, mask)
    
    colors = ['turquoise', 'orangered', 'orange', 'violet']
    #arrow_pts = [mitral_LA[0], tricus_RA[0], atrsep_LA[0], vensep_LV[0]]
    arrow_pts = [mitral_LA[0], tricus_RA[0], asep_pts, vsep_pts]
    #draw_arrows(mask, arrow_pts, colors=colors)
    
    sep_mask = np.copy(mask)
    sep_mask[np.where(asep_mask)] = 5
    sep_mask[np.where(vsep_mask)] = 6
    
    #plt.figure(figsize=(12,12))
    #plt.imshow(sep_mask)
    #plt.show()
    return sep_mask, arrow_pts


def find_contour_septal(contour, peak_pt, valve_pt, sep_cc):
    if np.ndim(valve_pt) == 2:
        valve_pt_distance = [get_pts_distance(sep_cc, pt_i) for pt_i in valve_pt]
        valve_pt = valve_pt[np.argmin(valve_pt_distance)]
        
    contour_img = np.zeros((256,256))
    contour_img[contour[:,0].astype(int), contour[:,1].astype(int)] = 1
    contour_img[int(valve_pt[0]), int(valve_pt[1])] = 0
    contour_img[int(peak_pt[0]), int(peak_pt[1])] = 0
    
    label = measure.label(contour_img)
    props = measure.regionprops(label)
    line_distance = [get_pts_distance(sep_cc, x.centroid) for x in props]
    line = props[np.argmin(line_distance)].coords
    pts = np.array([peak_pt, valve_pt])
    
    return pts, line
    
    
def process_mask2(mask):
    """
    Use end_points as peak point, and 
    use valve points from valve line
    """
    #print(f"processing {pid} .. ")
    centers = get_centers(mask)
    contours = get_contours(mask)
    end_pts = find_mask_end_pts(mask)
    
    mitral_LA = find_contour_contact(contours[0], contours[3])
    mitral_LV = find_contour_contact(contours[3], contours[0])
    tricus_RA = find_contour_contact(contours[1], contours[2])
    tricus_RV = find_contour_contact(contours[2], contours[1])
    
    atrsep_LA = find_contour_septal(contours[0], end_pts[0], mitral_LA[0], centers[1])
    atrsep_RA = find_contour_septal(contours[1], end_pts[1], tricus_RA[0], centers[0])
    vensep_LV = find_contour_septal(contours[3], end_pts[3], mitral_LV[0], centers[2])
    vensep_RV = find_contour_septal(contours[2], end_pts[2], tricus_RV[0], centers[3])
    #atrsep_LA = find_contour_contact(contours[0], contours[1])
    #atrsep_RA = find_contour_contact(contours[1], contours[0])
    #vensep_LV = find_contour_contact(contours[3], contours[2])
    #vensep_RV = find_contour_contact(contours[2], contours[3])
    
    asep_mask, asep_pts = get_septal_mask(atrsep_LA, atrsep_RA, mask)
    vsep_mask, vsep_pts = get_septal_mask(vensep_LV, vensep_RV, mask)
    
    colors = ['turquoise', 'orangered', 'orange', 'violet']
    #arrow_pts = [mitral_LA[0], tricus_RA[0], atrsep_LA[0], vensep_LV[0]]
    arrow_pts = [mitral_LA[0], tricus_RA[0], asep_pts, vsep_pts]
    #draw_arrows(mask, arrow_pts, colors=colors)
    
    sep_mask = np.copy(mask)
    sep_mask[np.where(asep_mask)] = 5
    sep_mask[np.where(vsep_mask)] = 6
    
    #plt.figure(figsize=(12,12))
    #plt.imshow(sep_mask)
    #plt.show()
    return sep_mask, arrow_pts

def find_contour_septal_LR(contourL, contourR, theta):
    theta = theta + np.pi/2
    rrL = contourL[:,1] * np.cos(theta) + contourL[:,0] * np.sin(theta)
    rrR = contourR[:,1] * np.cos(theta) + contourR[:,0] * np.sin(theta)
    ccL = np.mean(contourL, axis=0)
    ccR = np.mean(contourR, axis=0)
    
    r_max = min(rrL.max(), rrR.max())
    r_min = max(rrL.min(), rrR.min())
    
    cL0 = np.argmin(np.abs(rrL - r_min))
    cL1 = np.argmin(np.abs(rrL - r_max))
    
    cR0 = np.argmin(np.abs(rrR - r_min))
    cR1 = np.argmin(np.abs(rrR - r_max))
    
    ptsL, lineL = find_contour_septal(contourL, contourL[cL0], contourL[cL1], ccR)
    ptsR, lineR = find_contour_septal(contourR, contourR[cR0], contourR[cR1], ccL)
    
    return (ptsL, lineL), (ptsR, lineR)

def process_mask3(mask):
    """
    Use the theta of line connecting two contour
    to push down on two ends of both contour
    and cut the contour into lines.
    """
    #print(f"processing {pid} .. ")
    contours = get_contours(mask)
    end_pts = find_mask_end_pts(mask)
    thetas = get_thetas(mask)
    
    mitral_LA = find_contour_contact(contours[0], contours[3])
    mitral_LV = find_contour_contact(contours[3], contours[0])
    tricus_RA = find_contour_contact(contours[1], contours[2])
    tricus_RV = find_contour_contact(contours[2], contours[1])
    
    atrsep_LA, atrsep_RA = find_contour_septal_LR(contours[0], contours[1], thetas[0])
    vensep_LV, vensep_RV = find_contour_septal_LR(contours[3], contours[2], thetas[2])
    #atrsep_LA = find_contour_contact(contours[0], contours[1])
    #atrsep_RA = find_contour_contact(contours[1], contours[0])
    #vensep_LV = find_contour_contact(contours[3], contours[2])
    #vensep_RV = find_contour_contact(contours[2], contours[3])
    
    asep_mask, asep_pts = get_septal_mask(atrsep_LA, atrsep_RA, mask)
    vsep_mask, vsep_pts = get_septal_mask(vensep_LV, vensep_RV, mask)
    
    colors = ['turquoise', 'orangered', 'orange', 'violet']
    #arrow_pts = [mitral_LA[0], tricus_RA[0], atrsep_LA[0], vensep_LV[0]]
    arrow_pts = [mitral_LA[0], tricus_RA[0], asep_pts, vsep_pts]
    #draw_arrows(mask, arrow_pts, colors=colors)
    
    sep_mask = np.copy(mask)
    sep_mask[np.where(asep_mask)] = 5
    sep_mask[np.where(vsep_mask)] = 6
    
    #plt.figure(figsize=(12,12))
    #plt.imshow(sep_mask)
    #plt.show()
    return sep_mask, arrow_pts

def process_mask4(mask):
    """
    Use peak point from close contact, 
    Use valve point from valve line.
    """
    #print(f"processing {pid} .. ")
    contours = get_contours(mask)
    centers = get_centers(mask)
    
    mitral_LA = find_contour_contact(contours[0], contours[3])
    mitral_LV = find_contour_contact(contours[3], contours[0])
    tricus_RA = find_contour_contact(contours[1], contours[2])
    tricus_RV = find_contour_contact(contours[2], contours[1])
    
    #atrsep_LA = find_contour_septal(contours[0], end_pts[0], mitral_LA[0], centers[1])
    #atrsep_RA = find_contour_septal(contours[1], end_pts[1], tricus_RA[0], centers[0])
    #vensep_LV = find_contour_septal(contours[3], end_pts[3], mitral_LV[0], centers[2])
    #vensep_RV = find_contour_septal(contours[2], end_pts[2], tricus_RV[0], centers[3])
    
    atrsep_LA = find_contour_contact2(contours[0], contours[1], mitral_LA[0], centers[1])
    atrsep_RA = find_contour_contact2(contours[1], contours[0], tricus_RA[0], centers[0])
    vensep_LV = find_contour_contact2(contours[3], contours[2], mitral_LV[0], centers[2])
    vensep_RV = find_contour_contact2(contours[2], contours[3], tricus_RV[0], centers[3])
    
    #atrsep_LA = find_contour_contact(contours[0], contours[1])
    #atrsep_RA = find_contour_contact(contours[1], contours[0])
    #vensep_LV = find_contour_contact(contours[3], contours[2])
    #vensep_RV = find_contour_contact(contours[2], contours[3])
    
    asep_mask, asep_pts = get_septal_mask(atrsep_LA, atrsep_RA, mask)
    vsep_mask, vsep_pts = get_septal_mask(vensep_LV, vensep_RV, mask)
    
    colors = ['turquoise', 'orangered', 'orange', 'violet']
    #arrow_pts = [mitral_LA[0], tricus_RA[0], atrsep_LA[0], vensep_LV[0]]
    arrow_pts = [mitral_LA[0], tricus_RA[0], asep_pts, vsep_pts]
    #draw_arrows(mask, arrow_pts, colors=colors)
    
    sep_mask = np.copy(mask)
    sep_mask[np.where(asep_mask)] = 5
    sep_mask[np.where(vsep_mask)] = 6
    
    #plt.figure(figsize=(12,12))
    #plt.imshow(sep_mask)
    #plt.show()
    return sep_mask, arrow_pts


def get_measurement_images(pid_f, process_method, 
                           mask_dir=".", zip_dir="zips", 
                           fig_dir="figs"):
    pid, frame = pid_f.split("_")
    frame = int(frame)

    colors = ['turquoise', 'orangered', 'orange', 'violet']

    mask = np.load(f"{mask_dir}/{pid}.npy")[frame]
    zip_file = f"{zip_dir}/{pid}.zip"
    series_description = "CINE_segmented_LAX_4Ch"
    image = load_zip_data(zip_file, series_description, return_array=True)[frame]

    rgb = gray2rgb(image)
    try:
        fig_path = f"{fig_dir}/{pid}_F{frame}"
        sep_mask, arrow_pts = process_method(mask)
        draw_arrows(mask, arrow_pts, colors=colors, figname=f"{fig_path}_length.png")
        save_masks(image, sep_mask, f"{fig_path}_septal_masked.png")
        
    except Exception as err:
        print(pid)
        print(err)



def get_measurements(pid, process_method,
                     mask_dir=".", zip_dir="zips",
                     out_dir="measurements"):
    titles = ["AtrSepMass", "VenSepMass", "AtrSepLen", "VenSepLen", "MitAnnLen", "TriAnnLen"]
    try:
        masks = np.load(f"{mask_dir}/{pid}.npy")
        zip_file = f"{zip_dir}/{pid}.zip"
        pixelspacing = load_zip_pixelspacing(zip_file, "CINE_segmented_LAX_4Ch", by_frame=True)

        
        df = pd.DataFrame(columns=titles)
        for mask, pix in zip(masks, pixelspacing):
            try:
                sep_mask, arrow_pts = process_method(mask)
                atrsep_mass = (sep_mask == 5).sum() * np.prod(pix)
                vensep_mass = (sep_mask == 6).sum() * np.prod(pix)
                atrsep_leng = get_pts_distance_with_pix(arrow_pts[2], pix)
                vensep_leng = get_pts_distance_with_pix(arrow_pts[3], pix)
                mitann_leng = get_pts_distance_with_pix(arrow_pts[0], pix)
                triann_leng = get_pts_distance_with_pix(arrow_pts[1], pix)
                measurements = [atrsep_mass, vensep_mass, atrsep_leng, vensep_leng, mitann_leng, triann_leng]
            except:
                measurements = [None for i in titles]
            df = df.append(dict(zip(titles, measurements)), ignore_index=True)

        df.to_csv(f"{out_dir}/{pid}.csv", index=False)
        return df
        
    except Exception as err:
        print(pid)
        print(err)

    
## The implemented mask processing methods
process_mask_methods = {1: process_mask1,
                        2: process_mask2,
                        3: process_mask3,
                        4: process_mask4}

def main(args):
    #root_dir = "/Users/kexiao/Data/mr_oc/mr/train"
    #mask_dir = "la_4ch_mask/super_u/4ch"
    #csv = pd.read_csv(f"{root_dir}/labels.csv")

    method = process_mask_methods[args.method]
    print(f"Method {method} is used...")


    p = multiprocessing.Pool(1)
    if args.savefig:
        get_measurements_main = partial(get_measurement_images, process_method=method,
                                        mask_dir=args.mask_dir, zip_dir=args.zip_dir,
                                        fig_dir=args.out_dir)
        secure_folder(args.out_dir)
        csv = pd.read_csv(args.csv)
        pids = [f"{row.PID}_{row.Frame}" for i, row in csv.iterrows()]
    else:
        get_measurements_main = partial(get_measurements, process_method=method, 
                                        mask_dir=args.mask_dir, zip_dir=args.zip_dir,
                                        out_dir=args.out_dir)
        secure_folder(args.out_dir)
        csv = pd.read_csv(args.csv)
        pids = csv.PID[args.start: args.start+args.number]
        print(f"Processing pids in csv [{args.start}:{args.start+args.number}]")

    p.map(get_measurements_main, pids)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-C", "--csv", type=str, default="test_data/labels.csv", help="CSV to read PID from.")
    argparser.add_argument("-S", "--start", type=int, default=0, help="starting point in csv")
    argparser.add_argument("-N", "--number", type=int, default=3, help="number of points in csv")
    argparser.add_argument("--method", type=int, default=1, help="the method to use [1,2,3,4].")
    argparser.add_argument("-M", "--mask_dir", type=str, default="test_data/masks", help="the folder to load masks and pixelspacing from.")
    argparser.add_argument("-Z", "--zip_dir", type=str, default="test_data/zips", help="the zip folder to load images from")
    argparser.add_argument("-O", "--out_dir", type=str, default="outputs", help="the folder to save measurements into")
    argparser.add_argument("--savefig", action="store_true", help="whether to save masks")


    args = argparser.parse_args()
    main(args)
    #main()











