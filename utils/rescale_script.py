import json
import os

import PIL.Image
import io
import labelme
import cv2
import base64


SCALE_RATIO = 2
DIRS = ['K4/', 'R7/', 'J3/', 'Q3/', 'Q5/', 'J1/', 'J8/', 'J0/', 'K0/', 'J4/', 'R0/', 'K0R/', 'R6/', 'J4R/']
ROOT_JSON_DIR = '/work/azstaszewska/Data/Final Labels/'
SCALED_JSON_DIR = '/work/azstaszewska/Data/Final Labels 50/'
ROOT_IMG_DIR = '/work/azstaszewska/Data/Final Images/'
SCALED_IMG_DIR = '/work/azstaszewska/Data/Final Images 50/'

#files = ["R6_merged_regions.json", "K5_merged_regions.json", "J4_merged_regions.json", "J3_merged_regions.json", "H7_merged_regions.json", "H5_merged_regions.json", "H4_merged_regions.json", "H0_merged_regions.json", "G9_merged_regions.json", "G8_merged_regions.json", "G0_merged_regions.json"]
#sets = ["G0", "G8", "G9", "H0", "H4", "H5", "H6R", "H7", "J3", "J4", "J4R", "K0R", "K1", "K5", "Q0", "Q4", "Q6", "R0", "R2", "R6", "R6R"]
sets = ["G7"]
'''
for dir in DIRS:
    for filename in os.listdir(ROOT_IMG_DIR + dir):
	img = cv2.imread(ROOT_IMG_DIR + dir + filename)
	scaled = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
	cv2.imwrite(SCALED_IMG_DIR + dir + filename, scaled)

	json_filename = filename[:-4]+".json"
        annotation_json = json.load(open(ROOT_JSON_DIR+dir+json_filename))
        shapes = []
        for label in annotation_json["shapes"]:
            label["points"] = [[SCALE_RATIO*t[0], SCALE_RATIO*t[1]] for t in label["points"]]
            shapes.append(label)
        annotation_json["shapes"] = shapes

        with open(SCALED_JSON_DIR+dir+json_filename, 'w') as outfile:
            json.dump(annotation_json, outfile)
'''

DEST = "/home/azstaszewska/Data/MS Data/Stitched Final/"
SOURCE = "/home/azstaszewska/Data/MS Data/Stitched 50/"
for s in sets:
    data = labelme.LabelFile.load_image_file(DEST+ s+".png")

    annotation_json = json.load(open(SOURCE+s+"_50.json"))
    shapes = []
    for label in annotation_json["shapes"]:
        label["points"] = [[SCALE_RATIO*t[0], SCALE_RATIO*t[1]] for t in label["points"]]
        shapes.append(label)
    annotation_json["shapes"] = shapes
    annotation_json['imageData'] = base64.b64encode(data).decode('utf-8')
    annotation_json['imagePath'] =DEST + s+".png"
    annotation_json['flags'] = {}
    annotation_json['version'] = "4.6.0"

    with open(DEST + s + ".json", 'w') as outfile:
        json.dump(annotation_json, outfile)
