from PIL import Image, ImageDraw
import os
import json
from shapely.geometry import Polygon
import cv2
from helpers import bounding_box, normalize_dimensions, normalize_classname
import numpy



color_dict = {'gas entrapment': (11, 11, 11), 'lack of fusion': (255, 0, 0), 'keyhole': (0, 255, 0), "other": (150, 150, 150)}

sets = ["H8"]#, "J3R"]# ]#"H3", "G7", "H6R","R6R", "R0", "K1", "Q4"]
IMG_DIR = '/home/azstaszewska/Data/Final data/Images/'
LABELS_DIR = '/home/azstaszewska/Data/Final data/Labels/'
'''
for s in sets:
    for f in os.listdir(IMG_DIR + "/"+ s):
        img = cv2.imread(IMG_DIR + s +"/"+f)
        if not os.path.exists(LABELS_DIR+ s +"/"+f[:-4]+".json"):
            continue
        ann = json.load(open(LABELS_DIR+ s +"/"+f[:-4]+".json"))

        for defect in ann["shapes"]:
            label = normalize_classname(defect["label"])
            xy = [tuple(x) for x in defect["points"]]
            cv2.polylines(img, [numpy.int32(xy)], True, color_dict[label], thickness=3)
        cv2.imwrite("/home/azstaszewska/Data/Stitched Images/annotated/" + f, img)
'''
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
'Q9',
'R0',
"R6",
'R6R',
'R7',
"G0",
"H0",
"Q0",
'Q6',
'R2'
]
for s in sets:
    img = cv2.imread("/home/azstaszewska/Data/MS Data/Stitched Final/"+s+".png")
    LABELS_PATH ="/home/azstaszewska/Data/MS Data/Stitched Final/"+s+".json"

    ann = json.load(open(LABELS_PATH))

    for defect in ann["shapes"]:
        label = normalize_classname(defect["label"])
        if label == None:
            continue
        xy = [tuple(x) for x in defect["points"]]
        cv2.polylines(img, [numpy.int32(xy)], True, color_dict[label], thickness=3)
    cv2.imwrite("/home/azstaszewska/Data/MS Data/Viz/"+s+"_viz.png", img)
