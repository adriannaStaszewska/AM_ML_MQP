#extract features from the polygons
from __future__ import generators
import shapely
import json
from shapely.geometry import MultiPoint, Polygon,  LineString
import csv
import math
import matplotlib.pyplot as plt
import itertools
import numpy as np
from ellipse import LsqEllipse
import seaborn as sns
import pandas as pd
from math import sqrt
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    return sqrt(min_feret_sq), sqrt(max_feret_sq)


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




headers = ["sample", "polygon", "class", "area", "perimeter", "major_axis", "minor_axis", "convex_hull","ch_area", "centroid_x", "centriod_y", "max_feret_diameter", "min_feret_diameter", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "circularity", "aspect_ratio", "roundness", "solidity", "roughness", "shape_factor", "convexity", "perimeter_to_area_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "laser_power", "scan_speed", "layer_thickness", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]

    #load data set
ann_path = "/home/azstaszewska/Data/Detectron full set/final/train_trim_v2_tight.json"
f_ann = open(ann_path, )
annotation_json = json.load(f_ann)

rows_df = []
id = 0

process_params_df = pd.read_csv('/home/azstaszewska/Data/MS Data/AlSi10Mg_T160C_MQP_studies.csv')

for f in annotation_json: #extract info for each file
    if "flip" in f["file_name"].lower() or "gauss" in f["file_name"].lower() or "pepper" in f["file_name"].lower():
        continue
    img_height = f["height"]
    img_width = f["width"]

    sample = f["file_name"].split("/")[-2][:2]

    sample_process = process_params_df.loc[process_params_df['Sample Label'] == sample]

    laser_power = sample_process.iloc[0][1]
    scan_speed= sample_process.iloc[0][2]
    layer_thickness = sample_process.iloc[0][3]
    hatch_spacing  = sample_process.iloc[0][4]
    sim_melt_pool_width = sample_process.iloc[0][5]
    linear_energy_density = sample_process.iloc[0][7]
    surface_energy_density = sample_process.iloc[0][8]
    volumetric_energy_density = sample_process.iloc[0][9]
    img = mpimg.imread(f["file_name"])


    #extract data por each pore
    for pore in f["annotations"]:
        masked_img = img[pore["bbox"][1]:pore["bbox"][3], pore["bbox"][0]:pore["bbox"][2]]  # crop image to size of bounding box
        imgplot = plt.imshow(masked_img)
        id+=1
        polygon = pore["segmentation"]
        class_id = pore["category_id"]
        area = pore["area"]
        new_poly = []
        for p in polygon:
            poly = [(p[i], p[i+1]) for i in range(0, len(p)-1, 2)]
            new_poly.append(Polygon(poly))

        poly = unary_union(new_poly)
        if poly.geom_type=='MultiPolygon':
            for p in list(poly):
                p = Polygon(p)
                x, y = p.exterior.xy
                x = [x_i -  pore["bbox"][0] for x_i in x]
                y = [y_i -  pore["bbox"][1] for y_i in y]
                plt.plot(x,y)
            plt.savefig('/home/azstaszewska/Data/MS Data/Viz/Pores/pore_'+str(id)+".png")
        plt.clf()

        max_a = -1
        max_poly = None
        if poly.geom_type=='MultiPolygon':
            for p in list(poly):
                p = Polygon(p)
                if p.area > max_a:
                    max_a = p.area
                    max_poly = p
        else:
            max_poly = poly

        x, y = max_poly.exterior.coords.xy
        polygon_new = list(zip(x, y))
        '''
        new_poly = []


        for p in polygon:
            poly = [(p[i], p[i+1]) for i in range(0, len(p)-1, 2)]
            new_poly.append(poly)
            #x, y = Polygon(poly).exterior.xy
            #plt.plot(x, y)
            #plt.savefig('poly.png')
        #colors = itertools.cycle(["r", "b", "g"])

        x, y = Polygon(new_poly[0]).exterior.xy
        #plt.plot(x, y, color=next(colors))
        #plt.savefig('poly_'+str(id)+'.png')
        #plt.clf()
        #id+=1

        if len(new_poly) > 1:
            print("Mulit region")
            for p in new_poly:
                print(p)
                x, y = Polygon(poly).exterior.xy
                plt.plot(x, y, color=next(colors))
            id+=1
            plt.savefig('poly'+str(id)+'.png')
            plt.clf()
            continue
        '''

        perimeter = Polygon(polygon_new).length

        conv_hull = MultiPoint(polygon_new).convex_hull
        conv_hull_area = conv_hull.area
        conv_hull_perimeter = conv_hull.length

        mbr_points = list(zip(*Polygon(polygon_new).minimum_rotated_rectangle.exterior.coords.xy))

            # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

            # get major/minor axis measurements
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)

        centroid_x = Polygon(polygon_new).centroid.x
        centroid_y = Polygon(polygon_new).centroid.y
        ch_x, ch_y = conv_hull.exterior.coords.xy
        ch_shape = list(zip(ch_x, ch_y))

        min_feret_diameter, max_feret_diameter = min_max_feret(ch_shape)
        cols, rows = [], []
        for p in polygon:
            for i in range(len(p)):
                if i % 2 == 0:
                    cols.append(p[i])
                else:
                    rows.append(p[i])

        min_col = max(min(cols)-1, 0)
        max_col = min(max(cols)+1, img_height)
        min_row = max(min(rows)-1, 0)
        max_row = min(max(rows)+1, img_width)

        bbox_x = min_col
        bbox_y = min_row
        bbox_h = max_col - min_col
        bbox_w = max_row - min_row

        bbox_ratio = bbox_h/bbox_w

        circularity = 4*math.pi*area/perimeter**2
        aspect_ratio = major_axis/minor_axis
        roundness = 4*area/(math.pi*major_axis**2)
        solidity = area/conv_hull_area
        roughness = conv_hull_perimeter/perimeter
        shape_factor = 4*math.pi*area/conv_hull_perimeter**2
        convexity = area/conv_hull_area
        perimeter_to_area_ratio = perimeter/area
        try:
            X = np.array(list(zip(x, y)))
            reg = LsqEllipse().fit(X)
            center, width, height, phi = reg.as_parameters()
            if (np.iscomplex(width)):
                continue
            else:
                center = (float(center[0]), float(center[1]))
                width = float(width)
                height = float(height)
                phi = float(phi)
            ellipse_major_axis = max(width, height)*2
            ellipse_minor_axis = min(width, height)*2
            rows_df.append([f["file_name"], new_poly[0], class_id, area, perimeter, major_axis, minor_axis, ch_shape, conv_hull_area, centroid_x, centroid_y, max_feret_diameter, min_feret_diameter, bbox_x, bbox_y, bbox_h, bbox_w, bbox_ratio, circularity, aspect_ratio, roundness, solidity, roughness, shape_factor, convexity, perimeter_to_area_ratio, ellipse_major_axis, ellipse_minor_axis, phi, center[0], center[1], laser_power, scan_speed, layer_thickness, hatch_spacing, sim_melt_pool_width, linear_energy_density, surface_energy_density, volumetric_energy_density])
        except IndexError:
            center, width, height, phi, ellipse_major_axis, ellipse_minor_axis = [0, 0], 0, 0, 0, 0, 0
            print(f["file_name"])
            print("Could not work out ellipse")


pd.set_option('display.max_columns', None)
#save to csv
with open('/home/azstaszewska/Data/MS Data/extracted_features_subsets_2.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(map(lambda x: x, rows_df))

df = pd.read_csv('/home/azstaszewska/Data/MS Data/extracted_features_subsets_2.csv')

print(df.describe())
corr = df.corr()

plt.figure(figsize=(60,48))
sns.set(font_scale=1.4)
sns.heatmap(corr, cmap="PiYG",annot=True, center=0)
plt.savefig("/home/azstaszewska/correlations_pearson_2.png")
plt.clf()
'''
corr_kendall = df.corr(method="kendall")

sns.heatmap(corr, cmap="Greens",annot=True)
plt.savefig("/home/azstaszewska/correlations_kendall.png")
plt.clf()

corr_spearman = df.corr(method="spearman")

sns.heatmap(corr, cmap="Greens",annot=True)
plt.savefig("/home/azstaszewska/correlations_spearman.png")
plt.clf()
'''
corr[corr < 1].unstack().transpose()\
    .sort_values( ascending=False)\
    .drop_duplicates().to_csv("/home/azstaszewska/correlations_2.csv")
