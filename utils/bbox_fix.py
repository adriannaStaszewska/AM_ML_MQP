import sys
from os.path import exists
import json
import os
import numpy as np


FILE = '/home/azstaszewska/Data/Detectron full set/final/train_trim_new_2.json'
FILE_OUT = '/home/azstaszewska/Data/Detectron full set/final/train_trim_v2_tight.json'

f_ann = open(FILE, )
annotation_json = json.load(f_ann)
dataset = []

for a in annotation_json:
	new_annotations = []
	for instance in a['annotations']:
		print(instance)
		points = instance["segmentation"]
		cols, rows = [], []
		for p in points:
			for i in range(len(p)):
				if i % 2 == 0:
					cols.append(p[i])
				else:
					rows.append(p[i])
		print(rows)
		min_col = max(min(cols)-1, 0)
		max_col = min(max(cols)+1, int(a["width"]))
		min_row = max(min(rows)-1, 0)
		max_row = min(max(rows)+1, int(a["height"]))

		instance["bbox"] = [min_col, min_row, max_col, max_row]

		new_annotations.append(instance)
	a['annotations'] = new_annotations
	dataset.append(a)


with open(FILE_OUT, 'w') as f_ann:  # write back to the JSON
    json.dump(dataset, f_ann, indent=2)
