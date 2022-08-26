#Script to change the format of the data
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


dataset = []
OUT_FILE = sys.argv[2]
IN_FILE = sys.argv[1]

ROOT_IMG_DIR = '/home/azstaszewska/Data/MS Data/Split/Images2/'
ROOT_ANN_DIR = '/home/azstaszewska/Data/MS Data/Split/Labels2/'

CLASSES = ['lack of fusion', 'keyhole']


skip = True #masks already extracted
#get set from the file
paths_file = open(IN_FILE, 'r')
img_dirs = paths_file.readlines()
img_dirs = [line.rstrip('\n') for line in img_dirs]
id = 0

for f in img_dirs:
    image_data = {}
    image_path = ROOT_IMG_DIR+f
    if not exists(image_path):
        print("DOESNT exist")
        print(image_path)
        continue

    print(image_path)
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    print(height, width)

    ann_path = ROOT_ANN_DIR+f[:-4]+".json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)

    print(annotation_json["shapes"])

    image_data["file_name"] = image_path
    image_data["image_id"] = id
    image_data["height"] = int(height)
    image_data["width"] = int(width)

    annotations = []
    i = 0

    for instance in annotation_json["shapes"]:
        area = 0
        new_instance = {}
        new_instance["is_crowd"] = 0
        new_instance['bbox_mode']= BoxMode.XYXY_ABS
        class_name = normalize_classname(instance['label'])
        if skip:
            new_instance["category_id"] = CLASSES.index(normalize_classname(class_name))
            new_instance["area"] = Polygon(instance["points"]).area
            new_instance["segmentation"] = [[]]
            for p in instance["points"]:
                new_instance["segmentation"][0].append(p[0])
                new_instance["segmentation"][0].append(p[1])
            x, y = zip(*instance["points"])
            new_instance["bbox"] = [min(x), min(y), max(x), max(y)]
            annotations.append(new_instance)
        else:
            print("HERE!")
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
                print(instance['points'][0][0], instance['points'][1][0])
                col_min, col_max = int(min(instance['points'][0][0], instance['points'][1][0])), int(
                    max(instance['points'][0][0], instance['points'][1][0]))
                row_min, row_max = int(min(instance['points'][0][1], instance['points'][1][1])), int(
                    max(instance['points'][0][1], instance['points'][1][1]))
                print(col_min, col_max, row_min, row_max)
                col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)

                new_instance['bbox'] = [col_min, row_min, col_max, row_max]
                print(new_instance['bbox'])

                masked_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
                #brightness_adj = cv2.addWeighted(masked_img,1.5,np.zeros(masked_img.shape, image.dtype),0,0)
                cropped_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cropped_img_gray, 100, 200)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


                # apply contour to image and fill
                if (col_max-col_min)*(row_max-row_min)>7000:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                elif (col_max-col_min)*(row_max-row_min)>4000:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


                dilated = cv2.dilate(edged, kernel)
                contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = np.zeros(masked_img.shape)
                color = [255, 255, 255]
                cv2.fillPoly(polygon, contours, color)
                print(image_path)

                #cv2.imwrite(image_path[:-4]+"_poly_"+str(i)+".png", masked_img)
                #cv2.imwrite(image_path[:-4]+"_mask_"+str(i)+".png", polygon)
                i+=1
                #polygon = np.zeros(masked_img.shape)
                #color = [255, 255, 255]
                #cv2.fillPoly(polygon, contours, color)
                # normalize polygon to all boolean values and insert into mask
                #polygon_bool = np.alltrue(polygon == color, axis=2)
                new_instance["segmentation"] = []

                area = sum([cv2.contourArea(c) for c in contours])

                if area < 300:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                    dilated = cv2.dilate(edged, kernel)
                    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = np.zeros(masked_img.shape)
                    color = [255, 255, 255]
                    cv2.fillPoly(polygon, contours, color)
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

                threshold = (0, 150, 200)

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
                #brightness_adj = cv2.addWeighted(masked_img,1.5,np.zeros(image.shape, image.dtype),0,0)
                cropped_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cropped_img_gray, 100, 200)
                '''
                # apply contour to image and fill
                if (col_max-col_min)*(row_max-row_min)>7000:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                elif (col_max-col_min)*(row_max-row_min)>4000:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                '''
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                dilated = cv2.dilate(edged, kernel)
                contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                polygon = np.zeros(masked_img.shape)
                color = [255, 255, 255]
                cv2.fillPoly(polygon, contours, color)
                #cv2.imwrite("/work/azstaszewska/Data/Annotations viz/"+image_path[:-4]+"_poly_"+str(i)+".png", masked_img)
                #cv2.imwrite("/work/azstaszewska/Data/Annotations viz/"+image_path[:-4]+"_mask_"+str(i)+".png", polygon)
                i+=1

                area = sum([cv2.contourArea(c) for c in contours])

                if area < 300:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                    dilated = cv2.dilate(edged, kernel)
                    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = np.zeros(masked_img.shape)
                    color = [255, 255, 255]
                    cv2.fillPoly(polygon, contours, color)
                    area = sum([cv2.contourArea(c) for c in contours])

                #polygon = np.zeros(masked_img.shape)
                #color = [255, 255, 255]
                #cv2.fillPoly(polygon, contours, color)

                # normalize polygon to all boolean values and insert into mask
                #polygon_bool = np.alltrue(polygon == color, axis=2)


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

            if area > 1000:
                #print(new_instance["segmentation"])
                new_instance["area"] = area
                new_instance["category_id"] = CLASSES.index(normalize_classname(class_name))
                annotations.append(new_instance)
                print(new_instance)


    f_ann.close()
    id += 1
    image_data["annotations"] = annotations
    dataset.append(image_data)

    arr_img = np.array(image, dtype=np.uint8)

    augs = T.AugmentationList([
    T.RandomFlip(prob=1, horizontal=True, vertical=False),
    T.RandomFlip(prob=1, horizontal=False, vertical=True),
    T.RandomBrightness(intensity_min = 0.7, intensity_max = 1.3)
    ])  # type: T.Augmentation

    input = T.AugInput(arr_img)

    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    new_image_path = image_path[:-4]+"_flip.png"
    cv2.imwrite(new_image_path, image_transformed)
    new_image_data = {}
    new_image_data["annotations"] = []
    for an in annotations:
        new_instance = {}
        new_instance["is_crowd"] = 0
        new_instance['bbox_mode']= BoxMode.XYXY_ABS
        new_instance["category_id"] = an["category_id"]
        new_instance["area"] = an["area"]
        new_instance["bbox"] = transform.apply_box(an["bbox"]).tolist()[0]
        poly = []
        for s in an["segmentation"]:
            polygon = []
            print(s)
            for k in range(0, int(len(s)/2)):
                d1 = width-s[2*k]
                d2 = height-s[2*k+1]
                polygon.append(d1)
                polygon.append(d2)
            poly.append(polygon)

        new_instance["segmentation"] = poly
        new_image_data["annotations"].append(new_instance)

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)
    dataset.append(new_image_data)

    #print(new_image_data)

    id += 1
    '''
    augs = T.AugmentationList([
    T.RandomFlip(prob=1, horizontal=False, vertical=True),
    T.RandomBrightness(intensity_min = 0.75, intensity_max = 1.3)
    ])  # type: T.Augmentation

    input = T.AugInput(arr_img)

    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    new_image_path = image_path[:-4]+"_VRFlip.png"
    cv2.imwrite(new_image_path, image_transformed)
    new_image_data = {}
    new_image_data["annotations"] = []
    for an in annotations:
        new_instance = {}
        new_instance["is_crowd"] = 0
        new_instance['bbox_mode']= BoxMode.XYXY_ABS
        new_instance["category_id"] = an["category_id"]
        new_instance["area"] = an["area"]
        new_instance["bbox"] = transform.apply_box(an["bbox"]).tolist()[0]
        poly = []
        for s in an["segmentation"]:
            polygon = []
            for k in range(0, int(len(s)/2)):
                d1 = s[2*k]
                d2 = height-s[2*k+1]
                polygon.append(d1)
                polygon.append(d2)
            poly.append(polygon)

        new_instance["segmentation"] = poly
        new_image_data["annotations"].append(new_instance)

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)

    id += 1
    dataset.append(new_image_data)
    '''
    augs = T.AugmentationList([
    T.RandomFlip(prob=1, horizontal=True, vertical=False),
    T.RandomBrightness(intensity_min = 0.75, intensity_max = 1.3)
    ])  # type: T.Augmentation

    input = T.AugInput(arr_img)

    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    new_image_path = image_path[:-4]+"_HRFlip.png"
    cv2.imwrite(new_image_path, image_transformed)
    new_image_data = {}
    new_image_data["annotations"] = []
    for an in annotations:
        new_instance = {}
        new_instance["is_crowd"] = 0
        new_instance["area"] = an["area"]
        new_instance['bbox_mode']= BoxMode.XYXY_ABS
        new_instance["category_id"] = an["category_id"]
        new_instance["bbox"] = transform.apply_box(an["bbox"]).tolist()[0]
        poly = []
        for s in an["segmentation"]:
            polygon = []
            for k in range(0, int(len(s)/2)):
                d1 = width-s[2*k]
                d2 = s[2*k+1]
                polygon.append(d1)
                polygon.append(d2)
            poly.append(polygon)

        new_instance["segmentation"] = poly
        new_image_data["annotations"].append(new_instance)

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)
    id += 1
    dataset.append(new_image_data)

    aug = imgaug.augmenters.GaussianBlur(sigma=(0.75, 1.25))
    image_io = imageio.imread(image_path)

    img_aug = aug(image=image_io)
    new_image_path = image_path[:-4]+"_Gaussblur.png"
    imageio.imwrite(new_image_path, img_aug)

    new_image_data = {}
    new_image_data["annotations"] = annotations

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)
    id += 1
    dataset.append(new_image_data)

    aug = imgaug.augmenters.SaltAndPepper(0.015)
    image_io = imageio.imread(image_path)

    img_aug = aug(image=image_io)
    new_image_path = image_path[:-4]+"_saltAndPepper.png"
    imageio.imwrite(new_image_path, img_aug)

    new_image_data = {}
    new_image_data["annotations"] = annotations

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)
    id += 1
    dataset.append(new_image_data)


'''

    augs = T.AugmentationList([
    T.RandomRotation(angle=[30,60]),
    T.RandomBrightness(intensity_min = 0.75, intensity_max = 1.25)
    ])  # type: T.Augmentation

    input = T.AugInput(arr_img)

    # Apply the augmentation:
    transform = augs(input)  # type: T.Transform
    image_transformed = input.image  # new image
    new_image_path = image_path[:-4]+"_Rotation.png"
    cv2.imwrite(new_image_path, image_transformed)
    new_image_data = {}
    new_image_data["annotations"] = []
    for an in annotations:
        new_instance = {}
        new_instance["is_crowd"] = 0
        new_instance['bbox_mode']= BoxMode.XYXY_ABS
        new_instance["category_id"] = an["category_id"]
        new_instance["bbox"] = transform.apply_box(an["bbox"]).tolist()[0]
        poly = []
        for s in an["segmentation"]:
            polygon = []
            for k in range(0, int(len(s)/2)):
                d1 = width-s[2*k]
                d2 = s[2*k+1]
                polygon.append(d1)
                polygon.append(d2)
            poly.append(polygon)

        new_instance["segmentation"] = poly
        new_image_data["annotations"].append(new_instance)

    new_image_data["file_name"] = new_image_path
    new_image_data["image_id"] = id
    new_image_data["height"] = int(height)
    new_image_data["width"] = int(width)
    id += 1
    dataset.append(new_image_data)
'''

with open(OUT_FILE, 'w') as f_ann:  # write back to the JSON
    json.dump(dataset, f_ann, indent=2)
