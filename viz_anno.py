#File to train with detectron2 model

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, DefaultPredictor
import json
import random
import os
import cv2
from pathlib import Path

for d in ["G7"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/home/azstaszewska/Data/MS Data/Sets/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['lack of fusion', 'keyhole'])
	MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [184,0,0]])



train_metadata = MetadataCatalog.get("dataset_G7")
dataset_dicts_train = DatasetCatalog.get("dataset_G7")
for d in dataset_dicts_train:
	img = cv2.imread("/home/azstaszewska/Data/MS Data/Stitched Final/G7.png")
	visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1, instance_mode=1)
	vis = visualizer.draw_dataset_dict(d)
	cv2.imwrite("/home/azstaszewska/test.png",vis.get_image()[:, :, ::-1])



'''
save_loc = "/home/azstaszewska/Data/Annotations viz/"
for d in ["train_full", "val_full"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/home/azstaszewska/Data/Detectron full set/final/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['lack of fusion', 'keyhole'])
	MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [184,0,0]])



train_metadata = MetadataCatalog.get("dataset_train_full")
dataset_dicts_train = DatasetCatalog.get("dataset_train_full")

#val_metadata = MetadataCatalog.get("dataset_val_aug")
#dataset_dicts_val = DatasetCatalog.get("dataset_val_aug")

#test_metadata = MetadataCatalog.get("dataset_test")
#dataset_dicts_test = DatasetCatalog.get("dataset_test")

for d in dataset_dicts_train:
    if "flip" not in d["file_name"].lower() and "blur" not in d["file_name"].lower() and "salt" not in d["file_name"].lower():
	    print(d["file_name"])
	    img = cv2.imread(d["file_name"])
	    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1, instance_mode=1)
	    vis = visualizer.draw_dataset_dict(d)
	    cv2.imwrite(save_loc+"Val/"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])


for d in dataset_dicts_val:
    if "flip" not in d["file_name"].lower():
	    img = cv2.imread(d["file_name"])
	    visualizer = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=1, instance_mode=1)
	    vis = visualizer.draw_dataset_dict(d)
	    cv2.imwrite(save_loc+"Val/"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

for d in dataset_dicts_test:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=1, instance_mode=1)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite(save_loc+"Test/"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

'''
'''
for d in ["broken"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/work/azstaszewska/Data/Detectron full set/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['small lack of fusion porosity', 'medium lack of fusion porosity', 'large lack of fusion porosity', 'keyhole porosity'])
	MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [0, 184, 178], [0, 184, 101], [184,0,0]])

broken_metadata = MetadataCatalog.get("dataset_broken")
dataset_dicts_broken = DatasetCatalog.get("dataset_broken")

for d in dataset_dicts_broken:
    if "flip" not in d["file_name"].lower():
	    img = cv2.imread(d["file_name"])
	    visualizer = Visualizer(img[:, :, ::-1], metadata=broken_metadata, scale=0.5, instance_mode=1)
	    vis = visualizer.draw_dataset_dict(d)
	    cv2.imwrite(save_loc+"Broken/"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

'''