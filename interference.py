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


class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)



out_path = "/home/azstaszewska/Data/Predictions/Test/"

for d in ["train_full_new_2", "val_full_new_2", "test_full_new_2"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/home/azstaszewska/Data/Detectron full set/final/"+d+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['lack of fusion', 'keyhole'])
	MetadataCatalog.get("dataset_"+d).set(thing_colors=[[0, 80, 184], [184,0,0]])




cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

cfg.INPUT.MAX_SIZE_TRAIN = 2500

cfg.DATASETS.TRAIN = ("dataset_train_full_new_2",)
cfg.DATASETS.TEST = ("dataset_val_full_new_2",)

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
cfg.MODEL.WEIGHTs = str("/home/azstaszewska/Models/final_model_full_adj_masks_1/model_final.pth")


trainer = CocoTrainer(cfg)  # create trainer object from cfg
trainer.resume_or_load(resume=True)  # load weights

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("dataset_val_full_new_2", output_dir="/home/azstaszewska/output/final_model_full_adj_masks_1")

val_loader = build_detection_test_loader(cfg, 'dataset_val_full_new_2')
inference_on_dataset(trainer.model, val_loader, evaluator)
print(evaluator.evaluate())

print("___________________VALID__________________________")

train_loader = build_detection_test_loader(cfg, 'dataset_train_full_new_2')
inference_on_dataset(trainer.model, train_loader, evaluator)

print(evaluator.evaluate())

print("___________________TEST__________________________")
train_loader = build_detection_test_loader(cfg, 'dataset_test_full_new_2')
inference_on_dataset(trainer.model, train_loader, evaluator)

print(evaluator.evaluate())


dataset_stuff =  DatasetCatalog.get("dataset_test_full_new_2")
metadata = MetadataCatalog.get("dataset_test_full_new_2")
rand_set = random.sample(dataset_stuff, 60)
for d in rand_set:

	im = cv2.imread(d["file_name"])

	visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=2, instance_mode=1)
	vis = visualizer.draw_dataset_dict(d)
	cv2.imwrite(out_path + "GT_"+d["file_name"].split("/")[-1],vis.get_image()[:, :, ::-1])

	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1],metadata=metadata, scale=2,instance_mode=1)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	cv2.imwrite(out_path + d["file_name"].split("/")[-1], out.get_image()[:, :, ::-1])
