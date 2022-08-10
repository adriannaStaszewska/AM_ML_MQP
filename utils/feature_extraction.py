#extract features from the polygons
from __future__ import generators
import shapely
from shapely.geometry import MultiPoint, Polygon, Point, LineString
from shapely.ops import unary_union

import csv
import math
from math import sqrt, atan2
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from numpy.linalg import eig, inv, svd, det
import pandas as pd
import itertools
import cv2
from scipy.spatial.distance import pdist
import ast

def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1


def min_max_feret(Points):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = [((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)]
    min_feret_sq, min_feret_pair = min(squared_distance_per_pair)
    max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
    return sqrt(min_feret_sq), sqrt(max_feret_sq), min_feret_pair, max_feret_pair


def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return diam, pair

def min_feret(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    min_feret_sq,pair = min([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return min_feret_sq, pair

def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    if det(S) == 0.0:
        return [-1]
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:, 0]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * (
        (c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    down2 = (b * b - a * c) * (
        (a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    )
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return atan2(2 * b, (a - c)) / 2

def fit_ellipse(x, y):
    """@brief fit an ellipse to supplied data points: the 5 params
        returned are:
        M - major axis length
        m - minor axis length
        cx - ellipse centre (x coord.)
        cy - ellipse centre (y coord.)
        phi - rotation angle of ellipse bounding box
    @param x first coordinate of points to fit (array)
    @param y second coord. of points to fit (array)
    """
    a = __fit_ellipse(x, y)
    if len(a) == 1 and a[0] == -1:
        raise Exception("Cannot extract an ellipse")
    centre = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M, m = ellipse_axis_length(a)
    # assert that the major axix M > minor axis m

    if m > M:
        M, m = m, M
    # ensure the angle is betwen 0 and 2*pi
    phi -= 2 * np.pi * int(phi / (2 * np.pi))
    return M, m, centre[0], centre[1], phi


s_row_min, s_row_max = 1200, 4000
s_col_min, s_col_max = 1200, 19600
ROOT_IMG_DIR = "/home/azstaszewska/Data/MS Data/Stitched Final/"

headers = ["sample", "polygon", "class", "area", "perimeter", "major_axis", "minor_axis", "convex_hull","ch_area", "centroid_x", "centroid_y", "max_feret_diameter", "min_feret_diameter", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "aspect_ratio", "roundness", "solidity", "shape_factor", "convexity", "pta_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio", "laser_power", "scan_speed", "layer_thickness", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density", "include", "boundry", "area_in", "pore_in", "min_rect", "min_feret", "max_feret"]

sample_headers = ["set", "area", "pore_area", "pore_area_keyhole", "pore_area_lof", "porosity", "mean_distance", "laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width", "mean_area", "mean_area_keyhole", "mean_area_lof", "n_keyhole", "n_lof", "mean_sf", "mean_sf_key", "mean_sf_lof", "porosity_binary", "porosity_keyhole", "porosity_lof"]
CLASSES = ['lack of fusion', 'keyhole']

sets = ['Q6', 'R6', 'G8', 'H6R', 'J3R', 'Q3']
'''
sets = [
'G7',
#'G8',
'G9',
"H3",
'H4',
'H4R',
#'H6R',
'H7',
'H9',
'J0',
'J1',
'J3',
#"J3R",
'J4',
'J4R',
'J5',
'J8',
'J9',
'K0R',
'K1',
'K4',
'K5',
#'Q3',
'Q4',
'Q5',
'Q9',
'R0',
#"R6",
'R6R',
'R7',
"G0",
"H0"#,
#'Q6'
]
'''

process_params_df = pd.read_csv('/home/azstaszewska/Data/MS Data/AlSi10Mg_T160C_MQP_studies.csv')
bounding_rect_sample = Polygon([[s_col_min, s_row_min], [s_col_min, s_row_max], [s_col_max, s_row_max], [s_col_max, s_row_min]])

rows_df = []
rows_samples = []
valid_pores = []

for s in sets:
    #load data set
    print(s)

    ann_path = "/home/azstaszewska/Data/MS Data/Sets/v2 fixed sets/" + s +".json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)

    img = cv2.imread("/home/azstaszewska/Data/MS Data/Stitched Final/" + s+".png")
    cv2.rectangle(img, pt1=(s_col_min, s_row_min), pt2=(s_col_max, s_row_max), color=(0,0,0), thickness=5)
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Stitched Final/" + s+"_subsection_overlay.png", img)

    masked_img = img[s_row_min:s_row_max, s_col_min:s_col_max] 
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Stitched Final/" + s+"_subsection.png", masked_img)


    sample_process = process_params_df.loc[process_params_df['Sample Label'] == s[:2]]
    laser_power = sample_process.iloc[0][1]
    scan_speed= sample_process.iloc[0][2]
    layer_thickness = sample_process.iloc[0][3]
    hatch_spacing  = sample_process.iloc[0][4]
    sim_melt_pool_width = sample_process.iloc[0][5]
    linear_energy_density = sample_process.iloc[0][7]
    surface_energy_density = sample_process.iloc[0][8]
    volumetric_energy_density = sample_process.iloc[0][9]

    new_annotations = []
    sample_pores = []

    #extract info for each pore
    for f in annotation_json:
        sample = s
        class_name = f["class_name"]
        if class_name == "area":
            continue
        class_id = CLASSES.index(class_name)
        points = f["polygon"]
        poly = Polygon(points)

        include, boundry = False, False
        a_in, p_in = 0, f['polygon']
        if bounding_rect_sample.contains(poly):
            include = True
        elif bounding_rect_sample.intersects(poly):
            boundry = True
            a_in = bounding_rect_sample.intersection(poly).area

        x, y = poly.exterior.coords.xy
        polygon_xy = list(zip(x, y))

        perimeter = poly.length
        area = poly.area

        #convex hull
        conv_hull = poly.convex_hull
        conv_hull_area = conv_hull.area
        conv_hull_perimeter = conv_hull.length

        mbr_points = list(zip(*Polygon(points).minimum_rotated_rectangle.exterior.coords.xy))

        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

        # get major/minor axis measurements
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)

        centroid_x = poly.centroid.x
        centroid_y =  poly.centroid.y
        ch_x, ch_y = conv_hull.exterior.coords.xy
        ch_shape = list(zip(ch_x, ch_y))

        min_feret_diameter, max_feret_diameter, min_feret, max_feret = min_max_feret(ch_shape)

        cols, rows = poly.exterior.coords.xy
        min_col, max_col = min(cols), max(cols)
        min_row, max_row = min(rows), max(rows)

        bbox_x = min_col
        bbox_y = min_row
        bbox_h = max_col - min_col
        bbox_w = max_row - min_row

        bbox_ratio = bbox_h/bbox_w

        shape_factor = 4*math.pi*area/perimeter**2
        aspect_ratio = major_axis/minor_axis
        roundness = 4*area/(math.pi*major_axis**2)
        solidity = area/conv_hull_area
        #roughness = conv_hull_perimeter/perimeter
        #shape_factor = 4*math.pi*area/conv_hull_perimeter**2
        convexity = area/conv_hull_area
        perimeter_to_area_ratio = perimeter/area
        try:
            ellipse_major_axis, ellipse_minor_axis, center_x, center_y, phi = fit_ellipse(np.array(x), np.array(y))
            ellipse_ratio = ellipse_major_axis/ellipse_minor_axis
        except:
            print("failed")
            continue

        if area > 1000:
            rows_df.append([s, polygon_xy, class_id, area, perimeter, major_axis, minor_axis, ch_shape, conv_hull_area, centroid_x, centroid_y, max_feret_diameter, min_feret_diameter, bbox_x, bbox_y, bbox_h, bbox_w, bbox_ratio,  aspect_ratio, roundness, solidity, shape_factor, convexity, perimeter_to_area_ratio, ellipse_major_axis, ellipse_minor_axis, phi, center_x, center_y, ellipse_ratio, laser_power, scan_speed, layer_thickness, hatch_spacing, sim_melt_pool_width, linear_energy_density, surface_energy_density, volumetric_energy_density, include, boundry, a_in, p_in, mbr_points, min_feret, max_feret])
            sample_pores.append([s, polygon_xy, class_id, area, perimeter, major_axis, minor_axis, ch_shape, conv_hull_area, centroid_x, centroid_y, max_feret_diameter, min_feret_diameter, bbox_x, bbox_y, bbox_h, bbox_w, bbox_ratio, aspect_ratio, roundness, solidity, shape_factor, convexity, perimeter_to_area_ratio, ellipse_major_axis, ellipse_minor_axis, phi, center_x, center_y, ellipse_ratio, laser_power, scan_speed, layer_thickness, hatch_spacing, sim_melt_pool_width, linear_energy_density, surface_energy_density, volumetric_energy_density, include, boundry, a_in, p_in, mbr_points, min_feret, max_feret])

    df_sample = pd.DataFrame(sample_pores, columns=headers)

    sample_area = (s_row_max-s_row_min)*(s_col_max-s_col_min)
    #cv2.polylines(new_img, [np.int32(areas[-1][1])], True, (0,0,255), thickness=6)
    img = cv2.imread("/home/azstaszewska/Data/MS Data/Stitched Final/"+s+".png",cv2.IMREAD_GRAYSCALE)

    masked_img = img[s_row_min:s_row_max, s_col_min:s_col_max]
    thresh = 128
    img_binary = cv2.threshold(masked_img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Stitched Final/"+s+"_bw.png",img_binary)
    count = np.sum(img_binary == 0)
    porosity_binary = count/sample_area
    '''
    row =  [s, sample_area, porosity_binary]
    polygon_mask = np.zeros(masked_img.shape, dtype=np.uint8)
    polygon_mask.fill(255)
    '''
    '''
    img_copy = img_with_border.copy()
    cv2.polylines(img_copy, [np.int32(areas[-1][1])], True,  (0, 0, 255), thickness=1)
    cv2.fillPoly(img_copy, [np.int32(areas[-1][1])],  (0, 0, 255))
    frame_overlay=cv2.addWeighted(img_with_border, 0.1, img_copy,0.1, gamma=0)
    img_with_border = frame_overlay

    cv2.imwrite("/home/azstaszewska/Data/MS Data/Viz/" + s +"_sample_area.png", frame_overlay)
    '''

    inside_pores = df_sample[df_sample["include"]]
    analysis_pores = df_sample[df_sample["include"] | df_sample["boundry"]]
    '''
    for p in analysis_pores.to_dict("records"):
        polygon = p["polygon"]
        if p["boundry"]:
            if len(p["pore_in"]) > 1:
                for pk in p["pore_in"]:
                    print(pk)
                    cv2.fillPoly(polygon_mask, np.int32([pk]), 0)
        else:
            cv2.fillPoly(polygon_mask, np.int32([polygon]), 0)
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Viz/" + s +"_sample.png", frame_overlay)
    '''
    inside_keyhole, inside_lof = pd.DataFrame(), pd.DataFrame()
    if len(inside_pores) != 0:
        inside_keyhole = inside_pores[inside_pores["class"]==CLASSES.index("keyhole")]
        inside_lof = inside_pores[inside_pores["class"]==CLASSES.index("lack of fusion")]

    overlap_pores = df_sample[df_sample["boundry"]]
    overlap_pore_area, overlap_k, overlap_lof =0,0,0
    if(len(overlap_pores )> 0):
        overlap_pore_area = overlap_pores["area_in"].sum()
        overlap_lof = overlap_pores[overlap_pores["class"]==CLASSES.index("keyhole")]['area_in'].sum()
        overlap_k = overlap_pores[overlap_pores["class"]==CLASSES.index("lack of fusion")]["area_in"].sum()


    mean_distance, mean_distance_keyhole, mean_distance_lof = None, None, None
    pore_area, mean_area, mean_sf =0, 0, None
    pore_area_key, mean_area_key, mean_sf_key =0, 0, None
    pore_area_lof, mean_area_lof, mean_sf_lof =0, 0, None


    if len(inside_pores) != 0:
        pore_area = inside_pores["area"].sum()
        mean_area = inside_pores["area"].mean()
        mean_sf= inside_pores["shape_factor"].mean()
        X = list(zip(inside_pores["centroid_x"], inside_pores["centroid_y"]))
        mean_distance = np.mean(pdist(X))

    n_keyhole = len(inside_keyhole)
    n_lof = len(inside_lof)

    if len(inside_keyhole) != 0:
        pore_area_key = inside_keyhole["area"].sum()
        mean_area_key = inside_keyhole["area"].mean()
        mean_sf_key = inside_keyhole["shape_factor"].mean()
        X = list(zip(inside_keyhole["centroid_x"], inside_keyhole["centroid_y"]))
        mean_distance_key = np.mean(pdist(X))
    if len(inside_lof) != 0:
        pore_area_lof = inside_lof["area"].sum()
        mean_area_lof = inside_lof["area"].mean()
        mean_sf_lof = inside_lof["shape_factor"].mean()
        X = list(zip(inside_lof["centroid_x"],inside_lof["centroid_y"]))
        mean_distance_lof = np.mean(pdist(X))

    porosity_keyhole = (overlap_k+pore_area_key)/sample_area
    porosity_lof = (overlap_lof+pore_area_lof)/sample_area


    row = [s, sample_area, (pore_area+overlap_pore_area), pore_area_key, pore_area_lof, (pore_area+overlap_pore_area)/sample_area, mean_distance, laser_power, scan_speed, hatch_spacing, sim_melt_pool_width, mean_area, mean_area_key, mean_area_lof, n_keyhole, n_lof, mean_sf, mean_sf_key, mean_sf_lof, porosity_binary, porosity_keyhole, porosity_lof]

    rows_samples.append(row)

#save to csv
df = pd.DataFrame(rows_df, columns=headers)
df.to_csv('/home/azstaszewska/Data/MS Data/extracted_features_test.csv', index=False)
len(df[df["include"]])
df_samples = pd.DataFrame(rows_samples, columns=sample_headers)
df_samples.to_csv('/home/azstaszewska/Data/MS Data/sample_summary_test.csv', index=False)
