import json
import json
import os

SCALE_RATIO = 0.5
DIRS = ['K4/', 'R7/', 'J3/', 'Q3/', 'Q5/', 'J1/', 'J8/', 'J0/', 'K0/', 'J4/', 'R0/', 'K0R/', 'R6/', 'J4R/'] 
ROOT_JSON_DIR = '/work/azstaszewska/Data/Final Labels/'
SCALED_JSON_DIR = '/work/azstaszewska/Data/Final Labels 50/'
ROOT_IMG_DIR = '/work/azstaszewska/Data/Final Images/'
SCALED_IMG_DIR = '/work/azstaszewska/Data/Final Images 50/'



for dir in DIRS:
    for filename in os.listdir(ROOT_IMG_DIR + dir):
	img = cv2.imread(ROOT_IMG_DIR + dir + filename)
	scaled = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
	cv2.imwrite(SCALED_IMG_DIR + dir + filename, scaled)

	json_filename = filename[:-4]+".json"
        annotation_json = json.load(open(ROOT_JSON_DIR+dir+json_filename))
        shapes = []
        for label in annotation_json["shapes"]:
            label["points"] = [[SCALE_RATIO*t[0], SCALE_RATIO*t[1]] for t in label["points"]]
            shapes.append(label)
        annotation_json["shapes"] = shapes

        with open(SCALED_JSON_DIR+dir+json_filename, 'w') as outfile:
            json.dump(annotation_json, outfile)
