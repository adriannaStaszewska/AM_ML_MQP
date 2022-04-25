import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import normalize_classname


IN_FILE_TRAIN = '/work/azstaszewska/train_set_trimmed.txt'#sys.argv[1] #txt in
IN_DATA_FILE_TRAIN = '/work/azstaszewska/Data/Detectron full set/train_aug_trimmed.json' #sys.argv[2] #JSON in
OUT_FILE_TRAIN = '/work/azstaszewska/Data/Detectron full set/train_aug_trimmed_2.json' #sys.argv[3] #where to save

IN_FILE_VAL = '/work/azstaszewska/val_set_trimmed.txt' #sys.argv[3] #txt in
IN_DATA_FILE_VAL = '/work/azstaszewska/Data/Detectron full set/val_aug_trimmed.json' #sys.argv[5] #JSON in
OUT_FILE_VAL = '/work/azstaszewska/Data/Detectron full set/val_aug_trimmed_2.json' #sys.argv[5] #where to save

ROOT_IMG_DIR = '/work/azstaszewska/Data/Full data/Images/'
ROOT_ANN_DIR = '/work/azstaszewska/Data/Full data/Labels/'

CLASSES = ['small lack of fusion porosity', 'medium lack of fusion porosity', 'large lack of fusion porosity', 'keyhole porosity']


with open(IN_DATA_FILE_TRAIN, 'r') as f_ann:  # read JSON
    annotation_json = json.load(f_ann)

paths_file = open(IN_FILE_TRAIN, 'r')
img_dirs = paths_file.readlines()
img_dirs = [line.rstrip('\n') for line in img_dirs]

trimmed_data = []
trimmed_data_train = []
trimmed_data_val = []
all_areas_keyhole = []
all_areas_LOF = []


paths_file = open("/work/azstaszewska/sets/H0_train.txt", 'r')
H0_train_list = paths_file.readlines()
H0_train_list = [line.rstrip('\n') for line in img_dirs]

paths_file = open("/work/azstaszewska/sets/H0_val.txt", 'r')
H0_val_list = paths_file.readlines()
H0_val_list = [line.rstrip('\n') for line in img_dirs]

img_dirs = img_dirs + H0_train_list


with open('/work/azstaszewska/Data/Detectron full set/train_H0.json', 'r') as f_ann:  # read JSON
    annotation_json_H0 = json.load(f_ann)


for img in annotation_json_H0:
    file_name = img["file_name"].replace(ROOT_IMG_DIR, "")[:-4]
    if (file_name in H0_train_list) or (file_name.replace("_saltAndPepper", "") in H0_train_list ) or (file_name.replace("_HRFlip", "") in H0_train_list) or (file_name.replace("_VRFlip","") in H0_train_list ) or (file_name.replace("_flip", "") in H0_train_list ) or (file_name.replace("_Gaussblur", "") in H0_train_list):
        annotation_json.append(img)

for img in annotation_json:
    new_annotations = []
    file_name = img["file_name"].replace(ROOT_IMG_DIR, "")[:-4]
    if (file_name in img_dirs) or (file_name.replace("_saltAndPepper", "") in img_dirs) or (file_name.replace("_HRFlip", "") in img_dirs) or (file_name.replace("_VRFlip","") in img_dirs) or (file_name.replace("_flip", "") in img_dirs) or (file_name.replace("_Gaussblur", "") in img_dirs):
        for a in img["annotations"]:
            if a["area"]>6000:
                if a["category_id"] == 3:
                    all_areas_keyhole.append(a["area"])
                else:
                    all_areas_LOF.append(a["area"])
                new_annotations.append(a)

        img['annotations'] = new_annotations
        trimmed_data.append(img)
        trimmed_data_train.append(img)

with open(OUT_FILE_TRAIN, 'w') as f_ann:  # write back to the JSON
    json.dump(trimmed_data_train, f_ann, indent=2)


with open(IN_DATA_FILE_VAL, 'r') as f_ann:  # read JSON
    annotation_json = json.load(f_ann)

paths_file = open(IN_FILE_VAL, 'r')
img_dirs = paths_file.readlines()
img_dirs = [line.rstrip('\n') for line in img_dirs]

img_dirs = img_dirs + H0_val_list

with open('/work/azstaszewska/Data/Detectron full set/val_H0.json', 'r') as f_ann:  # read JSON
    annotation_json_H0 = json.load(f_ann)

for img in annotation_json_H0:
    file_name = img["file_name"].replace(ROOT_IMG_DIR, "")[:-4]
    if (file_name in H0_val_list) or (file_name.replace("_saltAndPepper", "") in H0_val_list ) or (file_name.replace("_HRFlip", "") in H0_val_list) or (file_name.replace("_VRFlip","") in H0_val_list ) or (file_name.replace("_flip", "") in H0_val_list ) or (file_name.replace("_Gaussblur", "") in H0_val_list):
        annotation_json.append(img)


for img in annotation_json:
    new_annotations = []
    file_name = img["file_name"].replace(ROOT_IMG_DIR, "")[:-4]
    print(file_name)
    if (file_name in img_dirs) or (file_name.replace("_saltAndPepper", "") in img_dirs) or (file_name.replace("_HRFlip", "") in img_dirs) or (file_name.replace("_VRFlip","") in img_dirs) or (file_name.replace("_flip", "") in img_dirs) or (file_name.replace("_Gaussblur", "") in img_dirs):
        for a in img["annotations"]:
            if a["area"]>6000:
                if a["category_id"] == 3:
                    all_areas_keyhole.append(a["area"])
                else:
                    all_areas_LOF.append(a["area"])
                new_annotations.append(a)

        img['annotations'] = new_annotations
        trimmed_data.append(img)
        trimmed_data_val.append(img)

with open(OUT_FILE_VAL, 'w') as f_ann:  # write back to the JSON
    json.dump(trimmed_data_val, f_ann, indent=2)

print("keyhole: " + str(len(all_areas_keyhole)))
print("lof: " + str(len(all_areas_LOF)))
plt.title("Distibution of pore areas for lack of fusion porosity")
_ = plt.hist(all_areas_LOF)
plt.xlabel("Size")
plt.ylabel("Frequency")
plt.savefig("lof_freq.png")

plt.title("Distibution of pore areas for keyhole porosity")
_ = plt.hist(all_areas_keyhole)
plt.savefig("keyhole_freq.png")


sorted_lof = sorted(all_areas_LOF)
print(all_areas_LOF)
bin_size = int(len(all_areas_LOF)/3)
print(f'Bin 1: {sorted_lof[0]}, {sorted_lof[bin_size]}')
print(f'Bin 2: {sorted_lof[bin_size]}, {sorted_lof[bin_size*2]}')
print(f'Bin 3: {sorted_lof[bin_size*2+1]}, {sorted_lof[-1]}')

bin_size = int(len(all_areas_LOF)/2)
print(f'Bin 1: {sorted_lof[0]}, {sorted_lof[bin_size]}')
print(f'Bin 2: {sorted_lof[bin_size]}, {sorted_lof[-1]}')
