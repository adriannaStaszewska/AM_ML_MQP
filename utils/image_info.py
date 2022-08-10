#extract image info
import cv2
import pandas as pd
import numpy as np
import csv
from PIL import Image, ImageOps
import shapely
from shapely.geometry import MultiPoint, Polygon, Point, LineString

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
#"Q0",
#"R2",
'Q6'
]

df_pores = pd.read_csv('/home/azstaszewska/Data/MS Data/extracted_features_new_2.csv', )

row_min, row_max = 512, 4300
col_min, col_max = 640, 19840
ROOT_IMG_DIR = "/home/azstaszewska/Data/MS Data/Stitched Final/"
rows = []
cols = ["set", "height", "width", "area", "pore_area", "porosity", "laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width"]
for s in sets:
    img = cv2.imread(ROOT_IMG_DIR+s + ".png")
    new_img = img[row_min:row_max, col_min:col_max]
    brightness_adj = cv2.addWeighted(new_img,1,np.zeros(new_img.shape, new_img.dtype),0,10)
    img_with_border = cv2.copyMakeBorder(new_img, 30, 30,30, 30, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.imwrite(ROOT_IMG_DIR +s+"_trim.png", img_with_border)
    img_gray = cv2.cvtColor(img_with_border, cv2.COLOR_BGR2GRAY)
    img_gray[img_gray<100] = 0
    edged = cv2.Canny(img_gray, 200, 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilated = cv2.dilate(edged, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    areas = [[cv2.contourArea(c), [[int(p[0][0]), int(p[0][1])] for p in c]] for c in contours]
    areas.sort(key=lambda x: x[0])
    sample_area = areas[-1][0]
    img_copy = img_with_border.copy()
    cv2.polylines(img_copy, [np.int32(areas[-1][1])], True,  (0, 0, 255), thickness=1)
    cv2.fillPoly(img_copy, [np.int32(areas[-1][1])],  (0, 0, 255))
    frame_overlay=cv2.addWeighted(img_with_border, 0.1, img_copy,0.1, gamma=0)
    img_with_border = frame_overlay

    cv2.imwrite("/home/azstaszewska/Data/MS Data/Viz/" + s +"_sample_area.png", frame_overlay)
    '''

    pores = df_pores[df_pores["sample"] == s]
    inside_pores = pores[(pores["bbox_x"] > col_min) & (pores["bbox_y"] > row_min) & ((pores["bbox_x"] + pores["bbox_w"]) < col_max) & ((pores["bbox_y"] + pores["bbox_h"]) < row_max)]
    pore_area = inside_pores["area"].sum()
    h, w, c = img.shape


    # print(pores.shape)
    # p = pores.sample()
    # laser_power = p["laser_power"]
    # scan_speed= p["scan_speed"]
    # hatch_spacing  = p["hatch_spacing"]
    # sim_melt_pool_width = p["sim_melt_pool_width"]

    rows.append([s, h, w, sample_area, pore_area, pore_area/sample_area])#, laser_power, scan_speed, hatch_spacing])

with open('/home/azstaszewska/Data/image_info.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(cols)
    write.writerows(rows)
