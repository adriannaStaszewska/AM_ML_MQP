#Generate train, validation and test sets
import os
import random

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.75, 0.2, 0.05
SETS_LOC = "/home/azstaszewska/sets/final/"
ROOT_IMG_DIR = '/home/azstaszewska/Data/MS Data/Split/Images2/'
ROOT_JSON_DIR = '/home/azstaszewska/Data/MS Data/Split/Labels2/'


#no-extreme data
SETS = [
'G7',
'G8',
'G9',
"H3",
'H4',
'H4R',
'H6R',
'H7',
'H9',
'J0',
'J1',
'J3',
"J3R",
'J4',
'J4R',
'J5',
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
'R7'
]

train = []
val = []
test = []

for s in SETS:
    print(os.listdir(ROOT_IMG_DIR + s + '/'))
    for f in os.listdir(ROOT_IMG_DIR + s + '/'):
        if not ("_flip" in f or "_VRflip" in f or "_HRflip" in f or "_Gaussblur" in f or "_saltAndPepper" in f):
            if os.path.exists(ROOT_JSON_DIR+s+"/"+f[:-4]+".json"):
                x = random.random()
                if x < TRAIN_RATIO:
                    train.append(s + '/'+f + '\n')
                elif x < TRAIN_RATIO+VAL_RATIO:
                    val.append(s + '/'+f + '\n')
                else:
                    test.append(s + '/'+f + '\n')

trim_file = "_trim_2.txt"
for s in ["train", "val", "test"]:
    f = open(SETS_LOC+s+trim_file, 'w')
    f.writelines(eval(s))
    f.close()


#full available data
EXTREME_SETS = ["G0", "H0", 'Q6']
for s in EXTREME_SETS:
    for f in os.listdir(ROOT_IMG_DIR + s + '/'):
        if not ("_flip" in f or "_VRflip" in f or "_HRflip" in f or "_Gaussblur" in f or "_saltAndPepper" in f):
            if os.path.exists(ROOT_JSON_DIR+s+"/"+f[:-4]+".json"):
                x = random.random()
                if x < TRAIN_RATIO:
                    train.append(s + '/'+f + '\n')
                elif x < TRAIN_RATIO+VAL_RATIO:
                    val.append(s + '/'+f + '\n')
                else:
                    test.append(s + '/'+f + '\n')


full_file = "_full_2.txt"
for s in ["train", "val", "test"]:
    f = open(SETS_LOC+s+full_file, 'w')
    f.writelines(eval(s))
    f.close()
