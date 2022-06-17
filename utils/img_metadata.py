import json
import base64
import os
import os.path as osp
import sys
from os.path import exists
import PIL.Image
import io
import labelme


ROOT_IMG_DIR = '/home/azstaszewska/Data/Full data/Images/'
ROOT_ANN_DIR = '/home/azstaszewska/Data/Full data/Labels/'
IN_FILE = "val_set.txt"

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


    ann_path = ROOT_ANN_DIR+f+".json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)
    if 'imageData' in annotation_json.keys():
        print("Already includes image data")
        print(image_path)
        continue
    data = labelme.LabelFile.load_image_file(image_path)
    #with open(image_path, mode='rb') as file:
        #img = file.read()
    annotation_json['imageData'] = base64.b64encode(data).decode('utf-8')
    annotation_json['flags'] = {}
    annotation_json['version'] = "4.6.0"
    annotation_json['imagePath'] =image_path

    with open(ann_path, 'w') as f_ann:  # write back to the JSON
        json.dump(annotation_json, f_ann, indent=2)
