'''
Removes 20X_YZ at the end of image names
'''

import cv2
import os

ROOT_DIR = '/work/azstaszewska/Data/Full data/Images/' # root directory where all JSONs are contained
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
'R7'] 

for dir in SUB_DIRS:
    dir_path = ROOT_DIR + dir + '/'
    for file in os.listdir(dir_path):
        img_path = ROOT_DIR + dir + '/' + file
        print(img_path)
        if img_path[-11:] == '_20X_YZ.tif':
            img = cv2.imread(img_path)
            os.remove(img_path)
            img_path = img_path[:-11] + img_path[-4:]
            cv2.imwrite(img_path, img)


