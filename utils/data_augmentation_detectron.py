#Script to augment data

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

IN_FILE = sys.argv[1]

ROOT_IMG_DIR = '/work/azstaszewska/Data/Full data/Images/'

paths_file = open(IN_FILE, 'r')
img_dirs = paths_file.readlines()
img_dirs = [line.rstrip('\n') for line in img_dirs]

for f in img_dirs:
    image_path = ROOT_IMG_DIR+f+".png"
    if not exists(image_path):
        image_path = ROOT_IMG_DIR+f+".tif"
    if not exists(image_path):
        print("DOESNT exist")
        print(image_path)
        continue

    print(image_path)
    image = cv2.imread(image_path)

    aug = imgaug.augmenters.GaussianBlur(sigma=(0.75, 1.5))
    image_io = imageio.imread(image_path)

    img_aug = aug(image=image_io)
    new_image_path = image_path[:-4]+"_Gaussblur.png"
    imageio.imwrite(new_image_path, img_aug)

    aug = imgaug.augmenters.SaltAndPepper(0.025)
    image_io = imageio.imread(image_path)

    img_aug = aug(image=image_io)
    new_image_path = image_path[:-4]+"_saltAndPepper.png"
    imageio.imwrite(new_image_path, img_aug)
