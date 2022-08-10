#information about all the pores in each set is extracted
import sys
from os.path import exists
import json
import os
import numpy as np
import cv2
from helpers import bounding_box, normalize_dimensions, normalize_classname
from detectron2.structures.boxes import BoxMode
from detectron2.data import transforms as T
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.geometry import box
import imageio
import imgaug
import math
import time


ROOT = "/home/azstaszewska/Data/MS Data/Stitched Final/"
OUT = "/home/azstaszewska/Data/MS Data/Sets/"
CLASSES = ['lack of fusion', 'keyhole']

sets = [
# 'G7',
# 'G8',
# 'G9',
# "H3",
# 'H4',
# 'H4R',
 'H5',
# 'H6R',
# 'H7',
 'H8',
# #'H9',
# 'J0',
# 'J1',
# 'J3',
# "J3R",
# 'J4',
# 'J4R',
# 'J5'#,
 'J7'#,
# 'J8',
# 'J9',
# 'K0',
# 'K0R',
# #'K1',
# 'K4',
# #'K5',
# 'Q3',
# 'Q4',
# 'Q9',
# #'R0',
# "R6",
# 'R6R',
# 'R7',
# "G0",
# "H0",
# "Q0",
# 'Q6'
]


def extract_contours(instance, image, masked_img):
    brightness_adj = cv2.addWeighted(masked_img,1.25,np.zeros(masked_img.shape, image.dtype),0,0)
    cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cropped_img_gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    dilated = cv2.dilate(edged, kernel)
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_approx, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #instance["contours_tree"] = [item.tolist() for item in contours_tree]
    #instance["hierarchy_tree"] = hierarchy_tree.tolist()
    instance["contours_external"] = [item.tolist() for item in contours]
    instance["contours_approx"] = [item.tolist() for item in contours_approx]

    polygon = np.zeros(masked_img.shape)
    color = [255, 255, 255]
    cv2.fillPoly(polygon, contours, color)
    return instance, contours

s = sets[int(sys.argv[1])]
print(s)
image_path = ROOT + s + ".png"
image = cv2.imread(image_path)
height = image.shape[0]
width = image.shape[1]

ann_path = ROOT+s+".json"
f_ann = open(ann_path, )
annotation_json = json.load(f_ann)

annotations = []

for instance in annotation_json["shapes"]:
    area = 0
    new_instance = {}
    if instance['label'] == "Region":
        continue
    
    class_name = normalize_classname(instance['label'])
    print(instance)
    if class_name == 'other' or class_name == 'gas entrapment':
        continue

    if  instance["shape_type"] == 'circle':
        instance['shape_type'] = 'rectangle'
        center, buffer = instance['points'][0], instance['points'][1]
        radius = math.sqrt((center[0]-buffer[0])**2+(center[1]-buffer[1])**2)
        new_points = [[center[0]-radius, center[1]-radius], [center[0]+radius, center[1]+radius]]
        instance["points"] = new_points

    if instance["shape_type"] == 'rectangle' and len(instance['points'])==2:
        # extract row and col data and crop image to annotation size
        col_min, col_max = int(min(instance['points'][0][0], instance['points'][1][0])), int(
            max(instance['points'][0][0], instance['points'][1][0]))
        row_min, row_max = int(min(instance['points'][0][1], instance['points'][1][1])), int(
            max(instance['points'][0][1], instance['points'][1][1]))
        col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)

        new_instance['bbox'] = [col_min, row_min, col_max, row_max]
        masked_img = image[new_instance["bbox"][1]:new_instance["bbox"][3], new_instance["bbox"][0]:new_instance["bbox"][2]]  # crop image to size of bounding box
        new_instance, contours = extract_contours(new_instance, image, masked_img)
        new_instance["segmentation"] = []

        area = sum([cv2.contourArea(c) for c in contours])

        for c in contours:
            if cv2.contourArea(c) > 100:
                new_c = [[int(p[0][0])+col_min, int(p[0][1])+row_min] for p in c]
                cnt = []
                for p in new_c:
                    cnt.append(p[0])
                    cnt.append(p[1])
                new_instance["segmentation"].append(cnt)


    elif instance["shape_type"] == 'polygon' or len(instance['points'])>2:
        points = []
        [points.append(coord) for coord in instance['points']]

        points = np.array(points, dtype=np.int32)
        polygon_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [points], (255, 255, 255))
        bbox = bounding_box(instance['points'])
        new_instance['bbox'] = [bbox[0][0], bbox[0][1],bbox[1][0], bbox[1][1]]
        col_min = min(bbox[0][0], bbox[1][0])
        row_min = min(bbox[0][1], bbox[1][1])
        col_max = max(bbox[0][0], bbox[1][0])
        row_max = max(bbox[0][1], bbox[1][1])
        # apply mask
        masked_img = cv2.bitwise_and(image, polygon_mask)
        black_pixels = np.where(
            (masked_img[:, :, 0] == 0) &
            (masked_img[:, :, 1] == 0) &
            (masked_img[:, :, 2] == 0)
        )

        threshold =  (0, 150, 200)
        g = image[:, :, 1]
        r = image[:, :, 2]
        r = r.flatten()
        r = [val for val in r if val > threshold[2]]
        avg_r = np.average(r)
        avg_b = 0
        g = g.flatten()
        g = [val for val in g if val > threshold[1]]
        avg_g = np.average(g)

        avg_color = (avg_b, avg_g, avg_r)
        masked_img[black_pixels] = avg_color
        new_instance, contours = extract_contours(new_instance, image, masked_img)
        #cv2.imwrite("/work/azstaszewska/Data/Annotations viz/"+image_path[:-4]+"_poly_"+str(i)+".png", masked_img)
        #cv2.imwrite("/work/azstaszewska/Data/Annotations viz/"+image_path[:-4]+"_mask_"+str(i)+".png", polygon)
        area = sum([cv2.contourArea(c) for c in contours])

        new_instance["segmentation"] = []
        for c in contours:
            if cv2.contourArea(c) > 100:
                new_c = [[int(p[0][0]), int(p[0][1])] for p in c]
                cnt = []
                for p in new_c:
                    cnt.append(p[0])
                    cnt.append(p[1])
                cnt.append(new_c[0][0])
                cnt.append(new_c[0][1])
                new_instance["segmentation"].append(cnt)

    if area > 500:
        new_instance["area"] = area
        new_instance["category_id"] = CLASSES.index(normalize_classname(class_name))
        annotations.append(new_instance)

with open(OUT+s+"_X.json", 'w') as f_ann:  # write back to the JSON
    print(annotations)
    json.dump(annotations, f_ann, indent=2)
