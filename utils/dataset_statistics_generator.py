import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import normalize_classname

ROOT_IMG_DIR = '/work/azstaszewska/Data/Full data/Images/'
ROOT_LABEL_DIR = '/work/azstaszewska/Data/Full data/Labels/'
DIRS = ['G0',
'G7',
'G8',
'G9',
'H0',
'H4',
'H5',
'H6',
'H7',
'H8',
'H9',
'J0',
'J1',
'J3',
'J4',
'J4R',
'J5',
'J7',
'J8',
'J9',
'K0',
'K0R',
'K1',
'K1R',
'K4',
'K5',
'Q0',
'Q3',
'Q4',
'Q5',
'Q6',
'Q9',
'R0',
'R2',
'R5',
'R6',
'R7']

data = {}


def normalize_dimensions(col_min, col_max, row_min, row_max):
    return max(col_min, 0), col_max, max(row_min, 0), row_max

for d in DIRS:
    image_count = 0
    data[d] = {'lack of fusion porosity': [], 'keyhole porosity': []}
    for filename in os.listdir(ROOT_IMG_DIR + d + "/"):
        image_count += 1
        ann_path = ROOT_LABEL_DIR + d + "/"+ filename[:-4]+".json"
        image_path = ROOT_IMG_DIR + d + "/" + filename

        image = cv2.imread(image_path)

        with open(ann_path, 'r') as f_ann:  # read JSON
            annotation_json = json.load(f_ann)

        for a in annotation_json["shapes"]:
            label = normalize_classname(a['label'])
            if label == 'lack of fusion porosity' or a['label'] == 'keyhole porosity':
                if a['shape_type'] == 'rectangle' or len(a['points']) == 2:
                    # extract row and col data and crop image to annotation size
                    col_min, col_max = int(min(a['points'][0][0], a['points'][1][0])), int(
                        max(a['points'][0][0], a['points'][1][0]))
                    row_min, row_max = int(min(a['points'][0][1], a['points'][1][1])), int(
                        max(a['points'][0][1], a['points'][1][1]))
                    col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)
                    masked_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
                    brightness_adj = cv2.addWeighted(masked_img,1.5,np.zeros(masked_img.shape, image.dtype),0,0)
                    cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(cropped_img_gray, 30, 200)

                    # apply contour to image and fill
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                    dilated = cv2.dilate(edged, kernel)
                    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    total_area = sum([cv2.contourArea(c) for c in contours])

                elif a['shape_type'] == 'polygon':
                    # generate mask from polygon points
                    points = []
                    [points.append(coord) for coord in a['points']]
                    points = np.array(points, dtype=np.int32)
                    polygon_mask = np.zeros(image.shape, dtype=np.uint8)
                    cv2.fillPoly(polygon_mask, [points], (255, 255, 255))

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
                    brightness_adj = cv2.addWeighted(masked_img,1.5,np.zeros(image.shape, image.dtype),0,0)
                    cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(cropped_img_gray, 30, 200)

                    # apply contour to image and fill
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                    dilated = cv2.dilate(edged, kernel)
                    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    total_area = sum([cv2.contourArea(c) for c in contours])
                class_type = label
                if total_area > 1500:
                    data[d][class_type].append(total_area)

    print(d + "  "+str(image_count))


new_df = pd.DataFrame.from_dict(data)
new_df.to_csv("area_dist.csv")

all_areas_keyhole = []
all_areas_LOF = []

for key in data.keys():
    all_areas_keyhole += data[key]["keyhole porosity"]
    all_areas_LOF += data[key]["lack of fusion porosity"]

string_text = f'Keyhole count: {len(all_areas_keyhole)}; LOF count: {len(all_areas_LOF)}'

print(string_text)

plt.title("Distibution of pore areas for lack of fusion porosity")
_ = plt.hist(all_areas_LOF)
plt.xlabel("Size")
plt.ylabel("Frequency")
plt.savefig("lof_freq.png")

plt.title("Distibution of pore areas for keyhole porosity")
_ = plt.hist(all_areas_keyhole)
plt.savefig("keyhole_freq.png")

sorted_lof = sorted(all_areas_LOF)
bin_size = int(len(all_areas_LOF)/3)
print(f'Bin 1: {sorted_lof[0]}, {sorted_lof[bin_size]}')
print(f'Bin 2: {sorted_lof[bin_size]}, {sorted_lof[bin_size*2]}')
print(f'Bin 3: {sorted_lof[bin_size*2+1]}, {sorted_lof[-1]}')
