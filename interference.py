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

out_path = "/work/azstaszewska/Data/Predictions/"

for d in ["val_augmented"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/work/azstaszewska/Data/Detectron full set/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['small lack of fusion porosity', 'medium lack of fusion porosity', 'large lack of fusion porosity', 'keyhole porosity'])


cfg = get_cfg() # initialize cfg object
cfg.MODEL.WEIGHTs = str("/work/azstaszewska/Models/detectron5/model_final.pth")

predictor = DefaultPredictor(cfg)
dataset_stuff =  DatasetCatalog.get("dataset_val_augmented")
metadata = MetadataCatalog.get("dataset_val")
i=0
for d in random.sample(dataset_stuff, 20):
	im = cv2.imread(d["file_name"])
	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1],metadata=metadata, scale=0.5)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	cv2.imwrite(out_path + d["file_name"], out.get_image()[:, :, ::-1])
	i+=1
