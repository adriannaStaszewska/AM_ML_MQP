import os
from PIL import Image, ImageDraw

LOC = "/home/azstaszewska/Data/Stitched Images/ImageJ sets/"
IMAGE_ROOT = '/home/azstaszewska/Data/Full data/Images/'
IMAGES_DIRS = [ "R0_L/"]

for set in IMAGES_DIRS:
    all_files = os.listdir(IMAGE_ROOT + set)

    pad = len(str(len(all_files)))
    all_files_new = []
    for filename in all_files:
        split = filename.split("_")
        if len(split) == 3:
            continue
        print(filename)
        if "_" in filename:
            x, y = filename.split("_")[1].split(".")[0][0], filename.split("_")[1].split(".")[0][1:]
        else:
            x, y = filename.split(".")[0][0], filename.split(".")[0][1:]
        if len(y) == 1:
            y = "0" + y
        all_files_new.append(x+y)

    result = result_list = [i for _,i in sorted(zip(all_files_new,all_files))]
    os.makedirs(LOC+set, exist_ok=False)
    id = 1
    for f in result:
        im = Image.open(IMAGE_ROOT+set+f)
        im.save(LOC + set + str(id).zfill(pad) + ".png")
        id += 1
