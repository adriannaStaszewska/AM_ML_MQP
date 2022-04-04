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

save_loc = "/work/azstaszewska/Data/Annotations viz/"
'''

#for d in ["train", "val", "test"]:
	#DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/work/azstaszewska/Data/Detectron full set/"+d+".json")))
	#MetadataCatalog.get("dataset_"+d).set(thing_classes=['small lack of fusion porosity', 'medium lack of fusion porosity', 'large lack of fusion porosity', 'keyhole porosity'])
	#MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [0, 184, 178], [0, 184, 101], [184,0,0]])



#train_metadata = MetadataCatalog.get("dataset_train_augme")
#dataset_dicts_broken = DatasetCatalog.get("dataset_train_augmented")

val_metadata = MetadataCatalog.get("dataset_val_augmented")
dataset_dicts_val = DatasetCatalog.get("dataset_val_augmented")

test_metadata = MetadataCatalog.get("dataset_test")
dataset_dicts_test = DatasetCatalog.get("dataset_test")

for d in dataset_dicts_broken:
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5, instance_mode=2)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite(save_loc+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

#for d in dataset_dicts_val:
   # img = cv2.imread(d["file_name"])
   # visualizer = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.5)
   # vis = visualizer.draw_dataset_dict(d)
   # cv2.imwrite(save_loc+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

#for d in dataset_dicts_test:
  #  img = cv2.imread(d["file_name"])
  #  visualizer = Visualizer(img[:, :, ::-1], metadata=test_metadata, scale=0.5)
  #  vis = visualizer.draw_dataset_dict(d)
  #  cv2.imwrite(save_loc+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

'''

for d in ["train_augmented"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/work/azstaszewska/Data/Detectron full set/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['small lack of fusion porosity', 'medium lack of fusion porosity', 'large lack of fusion porosity', 'keyhole porosity'])
	MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [0, 184, 178], [0, 184, 101], [184,0,0]])

broken_metadata = MetadataCatalog.get("dataset_train_augmented")
dataset_dicts_broken = DatasetCatalog.get("dataset_train_augmented")

for d in dataset_dicts_broken:
    if "flip" not in lower(d["file_name"]):
	    img = cv2.imread(d["file_name"])
	    visualizer = Visualizer(img[:, :, ::-1], metadata=broken_metadata, scale=0.5, instance_mode=1)
	    vis = visualizer.draw_dataset_dict(d)
	    cv2.imwrite(save_loc+"Train/"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])
