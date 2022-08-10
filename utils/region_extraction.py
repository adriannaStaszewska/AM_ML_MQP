import json
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.validation import make_valid
from shapely.geometry import box
import cv2
import numpy as np
import math
import os
import labelme
import base64

ROOT_IMG_DIR ="/home/azstaszewska/Data/MS Data/Stitched Final/"
IMG_OUT_DIR = '/home/azstaszewska/Data/MS Data/Split/Images/'
LABELS_OUT_DIR = '/home/azstaszewska/Data/MS Data/Split/Labels/'

sets_name = ["G0", "G8", "G9", "H0", "H4", "H5", "H6R", "H7", "J3", "J4", "J4R", "K0R", "K1", "K5", "Q0", "Q4", "Q6", "R0", "R2", "R6", "R6R", "G7", "H3"]
stitched = True
SETS = [
'G7',
'G8',
'G9',
"H3",
'H4',
'H4R',
'H5',
'H6R',
'H7',
'H8',
'H9',
'J0',
'J1',
'J3',
"J3R",
'J4',
'J4R',
'J5',
'J7',
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


def normalize_dimensions(col_min, col_max, row_min, row_max):
    return max(col_min, 0), col_max, max(row_min, 0), row_max

if stitched: #to be used on stitched images

    for set in sets_name:
        os.makedirs(LABELS_OUT_DIR+set, exist_ok=True)
        os.makedirs(IMG_OUT_DIR+set, exist_ok=True)

        annotation_json = json.load(open(ROOT_IMG_DIR+set+".json"))
        img = cv2.imread(ROOT_IMG_DIR+set + ".png")
        if img is None:
            img = cv2.imread(ROOT_IMG_DIR+set+ '.tif')
        shapes = []
        regions = []
        instances = []
        for label in annotation_json["shapes"]:
            if label["label"] == "Region":
                regions.append(label)
            else:
                instances.append(label)

        for i in range(len(regions)):
            new_annot = {"shapes": []}
            r = regions[i]
            region = box(0,0,0,0) # to be overwritten
            if r['shape_type'] == 'rectangle':
                col_min, col_max = int(min(r['points'][0][0], r['points'][1][0])), int(
                    max(r['points'][0][0], r['points'][1][0]))
                row_min, row_max = int(min(r['points'][0][1], r['points'][1][1])), int(
                    max(r['points'][0][1], r['points'][1][1]))
                col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)
                cropped_img = img[row_min:row_max, col_min:col_max]
                region = box(r["points"][0][0], r["points"][0][1], r["points"][1][0], r["points"][1][1])
            elif r['shape_type'] == 'polygon':
                points = []
                [points.append(coord) for coord in r['points']]
                x_coords = [point[0] for point in points]
                col_min, col_max = int(min(x_coords)), int(max(x_coords))
                y_coords = [point[1] for point in points]
                row_min, row_max = int(min(y_coords)), int(max(y_coords))

                col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)

                cropped_img = img[row_min:row_max, col_min:col_max]
                points = [(point[0] - col_min, point[1] - row_min) for point in points] # adjust coords of points
                points = np.array(points, dtype=np.int32)

                polygon_mask = np.zeros(cropped_img.shape, dtype=np.uint8)
                cv2.fillPoly(polygon_mask, [points], (255, 255, 255))

                # apply mask
                cropped_img = cv2.bitwise_and(cropped_img, polygon_mask)
                black_pixels = np.where(
                    (cropped_img[:, :, 0] == 0) &
                    (cropped_img[:, :, 1] == 0) &
                    (cropped_img[:, :, 2] == 0)
                )
                cropped_img[black_pixels] = (0, 0, 0)

                region = box(col_min, row_min, col_max, row_max)

            shapes = []
            for l in instances:
                if l["shape_type"] == "polygon":
                    instance = Polygon(l["points"])
                elif l["shape_type"] == "rectangle":
                    instance = box(l["points"][0][0], l["points"][0][1], l["points"][1][0], l["points"][1][1])
                elif l['shape_type'] == 'circle':
                    center = (l["points"][0][0], l["points"][0][1])
                    buffer = math.sqrt((center[0]-l["points"][1][0])**2+(center[1]-l["points"][1][1])**2)
                    instance = Point(center).buffer(buffer)
                if instance.intersects(region) and (make_valid(instance).intersection(make_valid(region)).area >
                                                    0.9*make_valid(instance).area):
                    new_ins = [[p[0]-col_min, p[1]-row_min] for p in l["points"]]
                    new_label = {}
                    if len(new_ins) == 2:
                    	new_ins2 = []
                    	new_ins2.append([new_ins[0][0], new_ins[0][1]])
                    	new_ins2.append([new_ins[0][0], new_ins[1][1]])
                    	new_ins2.append([new_ins[1][0], new_ins[1][1]])
                    	new_ins2.append([new_ins[1][0], new_ins[0][1]])
                    	new_ins = new_ins2
                    new_label["points"] = new_ins
                    print(new_label["points"])
                    new_label["label"] = l["label"]
                    new_label["shape_type"] = l["shape_type"]
                    shapes.append(new_label)
            new_annot["shapes"] = shapes
            cv2.imwrite(IMG_OUT_DIR+set+"/"+set+'_'+ str(i)+'.png', cropped_img)
            data = labelme.LabelFile.load_image_file(IMG_OUT_DIR+set+"/"+set+'_'+ str(i)+'.png')
            new_annot['imageData'] = base64.b64encode(data).decode('utf-8')
            new_annot['flags'] = {}
            new_annot['version'] = "4.6.0"
            new_annot['imagePath'] =IMG_OUT_DIR+set+"/"+set+'_'+ str(i)+'.png'

            cv2.imwrite(IMG_OUT_DIR+set+"/"+set+'_'+ str(i)+'.png', cropped_img)

            with open(LABELS_OUT_DIR+set+"/"+set+'_'+ str(i)+".json", 'w') as outfile:
                json.dump(new_annot, outfile)
else:
    for set in SETS:
        for img in os.listdir(IMG_OUT_DIR+set):
            if ("_flip" in img or "_VRflip" in img or "_VRFlip" in img or "_HRflip" in img or "_HRFlip" in img  or "_Gaussblur" in img  or "_saltAndPepper" in img ):
                continue
            if not os.path.exists(LABELS_OUT_DIR+set+"/"+img[:-4]+".json"):
                continue
            annotation_json = json.load(open(LABELS_OUT_DIR+set+"/"+img[:-4]+".json"))
            img_f = cv2.imread(IMG_OUT_DIR+set+"/"+img)
            shapes = []
            regions = []
            instances = []
            for label in annotation_json["shapes"]:
                if label["label"] == "Region":
                    regions.append(label)
                else:
                    instances.append(label)
            if len(regions) is not 0:
                print("HAS REGIONS")
                print(img)
                print(regions)
                for i in range(len(regions)):
                    new_annot = {"shapes": []}
                    r = regions[i]
                    region = box(0,0,0,0) # to be overwritten
                    if r['shape_type'] == 'rectangle':
                        col_min, col_max = int(min(r['points'][0][0], r['points'][1][0])), int(
                            max(r['points'][0][0], r['points'][1][0]))
                        row_min, row_max = int(min(r['points'][0][1], r['points'][1][1])), int(
                            max(r['points'][0][1], r['points'][1][1]))
                        col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)
                        cropped_img = img_f[row_min:row_max, col_min:col_max]
                        region = box(r["points"][0][0], r["points"][0][1], r["points"][1][0], r["points"][1][1])
                    elif r['shape_type'] == 'polygon':
                        points = []
                        [points.append(coord) for coord in r['points']]
                        x_coords = [point[0] for point in points]
                        col_min, col_max = int(min(x_coords)), int(max(x_coords))
                        y_coords = [point[1] for point in points]
                        row_min, row_max = int(min(y_coords)), int(max(y_coords))

                        col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)

                        cropped_img = img_f[row_min:row_max, col_min:col_max]
                        points = [(point[0] - col_min, point[1] - row_min) for point in points] # adjust coords of points
                        points = np.array(points, dtype=np.int32)

                        polygon_mask = np.zeros(cropped_img.shape, dtype=np.uint8)
                        cv2.fillPoly(polygon_mask, [points], (255, 255, 255))

                        # apply mask
                        cropped_img = cv2.bitwise_and(cropped_img, polygon_mask)
                        black_pixels = np.where(
                            (cropped_img[:, :, 0] == 0) &
                            (cropped_img[:, :, 1] == 0) &
                            (cropped_img[:, :, 2] == 0)
                        )
                        cropped_img[black_pixels] = (0, 0, 0)

                        region = box(col_min, row_min, col_max, row_max)

                    shapes = []
                    for l in instances:
                        if l["shape_type"] == "polygon":
                            instance = Polygon(l["points"])
                        elif l["shape_type"] == "rectangle":
                            instance = box(l["points"][0][0], l["points"][0][1], l["points"][1][0], l["points"][1][1])
                        elif l['shape_type'] == 'circle':
                            center = (l["points"][0][0], l["points"][0][1])
                            buffer = math.sqrt((center[0]-l["points"][1][0])**2+(center[1]-l["points"][1][1])**2)
                            instance = Point(center).buffer(buffer)
                        if instance.intersects(region) and (make_valid(instance).intersection(make_valid(region)).area >
                                                            0.9*make_valid(instance).area):
                            new_ins = [[p[0]-col_min, p[1]-row_min] for p in l["points"]]
                            new_label = {}
                            if len(new_ins) == 2:
                            	new_ins2 = []
                            	new_ins2.append([new_ins[0][0], new_ins[0][1]])
                            	new_ins2.append([new_ins[0][0], new_ins[1][1]])
                            	new_ins2.append([new_ins[1][0], new_ins[1][1]])
                            	new_ins2.append([new_ins[1][0], new_ins[0][1]])
                            	new_ins = new_ins2
                            new_label["points"] = new_ins
                            print(new_label["points"])
                            new_label["label"] = l["label"]
                            new_label["shape_type"] = l["shape_type"]
                            shapes.append(new_label)
                    new_annot["shapes"] = shapes
                    cv2.imwrite(IMG_OUT_DIR+set+"/"+img, cropped_img)
                    data = labelme.LabelFile.load_image_file(IMG_OUT_DIR+set+"/"+img)
                    new_annot['imageData'] = base64.b64encode(data).decode('utf-8')
                    new_annot['flags'] = {}
                    new_annot['version'] = "4.6.0"
                    new_annot['imagePath'] =IMG_OUT_DIR+set+"/"+img

                    with open(LABELS_OUT_DIR+set+"/"+img[:-4]+".json", 'w') as outfile:
                        json.dump(new_annot, outfile)
