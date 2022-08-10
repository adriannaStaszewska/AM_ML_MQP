
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
import sys

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

for d in ["train_trim_1", "val_trim_1"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/home/azstaszewska/Data/Detectron full set/final/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['lack of fusion', 'keyhole'])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

cfg.INPUT.MAX_SIZE_TRAIN = 2500

cfg.DATASETS.TRAIN = ("dataset_train_trim_1",)
cfg.DATASETS.TEST = ("dataset_val_trim_1",)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 200
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.BASE_LR = 0.0001

cfg.MODEL.DEVICE='cuda'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 20
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

cfg.TEST.DETECTIONS_PER_IMAGE = 50
#cfg.TEST.EVAL_PERIOD = 100
print("LR: " + str(cfg.SOLVER.BASE_LR))


# model weights will be downloaded if they are not present
weights_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
if os.path.exists(weights_path):
    print('Using locally stored weights: {}'.format(weights_path))
else:
    weights_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    print('Weights not found, weights will be downloaded from source: {}'.format(weights_path))
cfg.MODEL.WEIGHTS = str(weights_path)
cfg.OUTPUT_DIR = "/home/azstaszewska/Models/final_model_trim_4"
# make the output directory
os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)

trainer = CocoTrainer(cfg)  # create trainer object from cfg
trainer.resume_or_load(resume=False)  # start training from iteration 0
trainer.train()  # train the model!

model_checkpoints = sorted(Path(cfg.OUTPUT_DIR).glob('*.pth'))  # paths to weights saved druing training
cfg.DATASETS.TEST = ("dataset_val_trim_1",)  # predictor requires this field to not be empty
cfg.MODEL.WEIGHTS = str(model_checkpoints[-1])  # use the last model checkpoint saved during training. If you want to see the performance of other checkpoints you can select a different index from model_checkpoints.
os.makedirs(Path("/home/azstaszewska/output/final_model_trim_4"), exist_ok=True)

predictor = DefaultPredictor(cfg)  # create predictor object
evaluator = COCOEvaluator("dataset_val_trim_1", output_dir="/home/azstaszewska/output/final_model_trim_4")

'''
with open("/work/azstaszewska/configs/config9.yaml", "w") as f:
  f.write(cfg.dump())
out_path = "/home/azstaszewska/Data/Predictions/"

for d in random.sample(dataset_stuff, 20):
	im = cv2.imread(d["file_name"])
	outputs = predictor(im)
	print(outputs)
	v = Visualizer(im[:, :, ::-1],metadata=metadata)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	out.save(out_path + d["file_name"].split("/")[-1])

'''
val_loader = build_detection_test_loader(cfg, 'dataset_val_trim_1')
inference_on_dataset(trainer.model, val_loader, evaluator)
print(evaluator.evaluate())

train_loader = build_detection_test_loader(cfg, 'dataset_train_trim_1')
inference_on_dataset(trainer.model, train_loader, evaluator)

