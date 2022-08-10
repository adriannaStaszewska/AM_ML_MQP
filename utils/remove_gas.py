#removes gas porosity instances

import sys
from os.path import exists
import json
import os
from helpers import bounding_box, normalize_dimensions, normalize_classname


ROOT = "/home/azstaszewska/Data/MS Data/Stitched Final/"
OUT = "/home/azstaszewska/Data/MS Data/Sets/"
CLASSES = ['lack of fusion', 'keyhole']

sets = [
# 'G7',
# 'G8',
# 'G9',
# "H3",
# 'H4',
# 'H4R',
 'H5',
# 'H6R',
# 'H7',
 'H8',
# #'H9',
# 'J0',
# 'J1',
# 'J3',
# "J3R",
# 'J4',
# 'J4R',
# 'J5'#,
 'J7'#,
# 'J8',
# 'J9',
# 'K0',
# 'K0R',
# #'K1',
# 'K4',
# #'K5',
# 'Q3',
# 'Q4',
# 'Q9',
# #'R0',
# "R6",
# 'R6R',
# 'R7',
# "G0",
# "H0",
# "Q0",
# 'Q6'
]


for s in sets:

    ann_path = ROOT+s+".json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)

    annotations = []

    for instance in annotation_json["shapes"]:

        area = 0
        new_instance = {}
        if instance['label'] == "Region" or instance["label"] == None:
            continue

        class_name = normalize_classname(instance['label'])
        if class_name == 'other' or class_name == 'gas entrapment':
            continue

        annotations.append(instance)
    annotation_json["shapes"] = annotations

    with open(ann_path, 'w') as f_ann:  # write back to the JSON
        json.dump(annotation_json, f_ann, indent=2)
