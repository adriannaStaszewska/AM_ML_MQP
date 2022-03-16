'''
JSON cleaner - removes unwanted classes
Replace SUB_DIRS with the appropriate annotation directories and you're set!
'''

import os
import json
import normalize_classname
import math

ROOT_DIR = '/work/azstaszewska/Data/Full data/Labels/' # root directory where all JSONs are contained
SUB_DIRS = ['G0',
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
'Q8',
'Q9',
'R0',
'R2',
'R5',
'R6',
'R7'] # subdirs containing JSONs

for dir in SUB_DIRS:
    dir_path = ROOT_DIR + dir + '/'
    for file in os.listdir(dir_path):
        ann_path = ROOT_DIR + dir + '/' + file
        print(ann_path)
        with open(ann_path, 'r') as f_ann: # read JSON
            annotation_json = json.load(f_ann)

        shapes = []
        for shape in annotation_json['shapes']:
            shape['label'] = normalize_classname.normalize_classname(shape['label'])
            if shape['shape_type'] == 'circle':
                shape['shape_type'] = 'rectangle'
                center, buffer = shape['points'][0], shape['points'][1]
                radius = math.sqrt((center[0]-buffer[0])**2+(center[1]-buffer[1])**2)
                new_points = [[center[0]-radius, center[1]-radius], [center[0]+radius, center[1]+radius]]
                shape["points"] = new_points
            if shape['label'] != 'gas entrapment porosity' or shape['label'] != 'other':
                shapes.append(shape)
        # rename filenames to remove 20X_YZ.json
        os.remove(ann_path) # delete old ann path
        if file[-12:] == '_20X_YZ.json':
            ann_path = ann_path[:-12] + ann_path[-5:] # remove 20X_YZ at the end of annotations

        annotation_json['shapes'] = shapes

        with open(ann_path, 'w') as f_ann: # write back to the JSON
            json.dump(annotation_json, f_ann, indent=2)
