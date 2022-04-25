import sys
from os.path import exists
import json
import os
import numpy as np


FILE = '/work/azstaszewska/Data/Detectron full set/val_aug_trimmed_2.json'
OUT_FILE = '/work/azstaszewska/Data/Detectron full set/val_aug_trimmed_2class_2.json'

f_ann = open(FILE, )
annotation_json = json.load(f_ann)
dataset = []

for a in annotation_json:
	new_annotations = []
	for instance in a['annotations']:
		class_id = instance["category_id"]
		if class_id == 3: #keyhole
			instance["category_id"] = 1
		else:
			instance["category_id"] = 0
		new_annotations.append(instance)
	a['annotations'] = new_annotations
	dataset.append(a)


with open(OUT_FILE, 'w') as f_ann:  # write back to the JSON
    json.dump(dataset, f_ann, indent=2)
