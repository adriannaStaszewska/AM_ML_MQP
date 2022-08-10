#Save extected polygons as labelme file
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
import labelme
import base64

sets = [
# 'G7',
# 'G8',
# 'G9',
# "H3",
# 'H4',
# 'H4R',
# #'H5',
# 'H6R',
# 'H7',
# #'H8',
# #'H9',***
#'J0',
# 'J1',
# 'J3',
# "J3R",
# 'J4',
# 'J4R',***
#'J5',
#'J7',
#'J8',
#'J9',
#'K0R',****
#'K1',
#'K4',****
'K5',
'Q3',
'Q4',
'Q5',
'Q9',
'R0',
"R6",
'R6R',
'R7',
"G0",
"H0",
"Q0",
"R2",
'Q6'
]


for s in sets:
    IMG = "/home/azstaszewska/Data/MS Data/Stitched Final/"+s+".png"
    ANN = "/home/azstaszewska/Data/MS Data/Sets/"+s+"_v2.json"
    OUT = "/home/azstaszewska/Data/MS Data/Sets/Labelme sets/v2/"

    CLASSES = ['lack of fusion', 'keyhole']

    image = cv2.imread(IMG)
    height = image.shape[0]
    width = image.shape[1]

    f_ann = open(ANN, )
    annotations = json.load(f_ann)
    annotation_json = {}
    annotation_json["shapes"] = []

    for ann in annotations:
        pore = {}
        pore["label"] = CLASSES[ann["category_id"]]
        pore["shape_type"] = "polygon"
        pore["flags"] = {}
        pore["group_id"] = None
        poly = []
        for p in ann["segmentation"]:
            poly_sub = [(p[i], p[i+1]) for i in range(0, len(p)-1, 2)]
            poly.append(poly_sub)

        max_a  = -1
        max_poly = None
        if len(poly) > 1:
            for p in list(poly):
                p = Polygon(p)
                if p.area > max_a:
                    max_a = p.area
                    max_poly = p
        else:
            max_poly = Polygon(poly[0])
        max_poly = max_poly.simplify(0.6)
        x, y = max_poly.exterior.coords.xy
        polygon_new = list(zip(x, y))
        pore["points"] = [list(p) for p in polygon_new]
        #print(pore)

        annotation_json["shapes"].append(pore)
        '''
        pore["label"] = "bbox"
        pore["shape_type"] = "rectangle"
        pore["points"] = [[ann["bbox"][0], ann["bbox"][1]], [ann["bbox"][2], ann["bbox"][3]]]
        annotation_json["shapes"].append(pore)
    	'''
    data = labelme.LabelFile.load_image_file(IMG)
    annotation_json['imageData'] = base64.b64encode(data).decode('utf-8')
    annotation_json['flags'] = {}
    annotation_json['version'] = "4.6.0"
    annotation_json['imagePath'] = IMG
    '''
    with open(OUT+s+".json", 'w') as f_ann:  # write back to the JSON
        json.dump(annotation_json, f_ann, indent=2)
    '''

    SCALE_RATIO = 0.5

    scaled = cv2.resize(image, (0, 0), fx=SCALE_RATIO , fy=SCALE_RATIO )
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Stitched Final/"+s+"_50.png", scaled)
    shapes = []
    for label in annotation_json["shapes"]:
        label["points"] = [[SCALE_RATIO*t[0], SCALE_RATIO*t[1]] for t in label["points"]]
        shapes.append(label)
    annotation_json["shapes"] = shapes
    data = labelme.LabelFile.load_image_file("/home/azstaszewska/Data/MS Data/Stitched Final/"+s+"_50.png")
    annotation_json['imageData'] = base64.b64encode(data).decode('utf-8')
    print(s)
    with open(OUT+s+"_50_v2.json", 'w') as f_ann:  # write back to the JSON
        json.dump(annotation_json, f_ann, indent=2)
