#Visialize entire set
import sys
from os.path import exists
import json
import os
import numpy as np
import cv2
from helpers import bounding_box, normalize_dimensions, normalize_classname
from shapely.ops import unary_union
from shapely.geometry import Polygon

SOURCE_FOLDER = "/home/azstaszewska/Data/MS Data/Sets/fixed sets/"
IMG_FOLDER = "/home/azstaszewska/Data/MS Data/Stitched Final/"
CLASSES = ['lack of fusion', 'keyhole']

sets = [
'G7',
'G8',
'G9',
"H3",
'H4',
'H4R',
#'H5',
'H6R',
'H7',
#'H8',
'H9',
'J0',
'J1',
'J3',
"J3R",
'J4',
'J4R',
'J5',
#'J7',
'J8',
'J9',
'K0',
'K0R',
'K1',
'K4',
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


color_dict = {'gas entrapment': (11, 11, 11), 'lack of fusion': (255, 0, 0), 'keyhole': (0, 255, 0), "other": (150, 150, 150)}
for s in sets:
    img = cv2.imread(IMG_FOLDER+s+".png")
    ann = json.load(open(SOURCE_FOLDER+ s +".json"))
    for defect in ann:
        img_copy = img.copy()
        label = normalize_classname(defect["class_name"])
        poly =Polygon(defect["polygon"])

        x, y = poly.exterior.coords.xy
        polygon = list(zip(x, y))
        cv2.polylines(img_copy, [np.int32(polygon)], True, color_dict[label], thickness=1)
        cv2.fillPoly(img_copy, [np.int32(polygon)], color_dict[label])
        frame_overlay=cv2.addWeighted(img, 0.6, img_copy,1-0.6, gamma=0)
        img = frame_overlay

    cv2.imwrite("/home/azstaszewska/Data/MS Data/Viz/" + s +"_new.png", frame_overlay)
