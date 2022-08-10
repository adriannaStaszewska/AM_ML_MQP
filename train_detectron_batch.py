
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
        os.makedirs("/home/azstaszewska/Batch models/coco_eval_"+slurm_task_id, exist_ok=True)
        output_folder = "/home/azstaszewska/Batch models/coco_eval_"+slurm_task_id

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

slurm_task_id = sys.argv[1]
val_data = json.load(open("/home/azstaszewska/Data/Detectron full set/final/val_trim_new_2.json"))
train_data = json.load(open("/home/azstaszewska/Data/Detectron full set/final/train_trim_new_2.json"))

data = val_data + train_data

train, val = [], []
i = 0
for img in data:
    if "flip" not in img["file_name"].lower() and "salt" not in img["file_name"].lower() and "blur" not in img["file_name"].lower():
        chance = random.random()
        if chance < 0.2:
            val.append(img)
        else:
            train = train + data[i:i+6]
    i+=1

random.shuffle(train)
random.shuffle(val)

#f = open("/home/azstaszewska/Data/Detectron full set/final/batch_temp/train_"+str(slurm_task_id)+".json", "x")

with open("/home/azstaszewska/Data/Detectron full set/final/batch_temp/train_"+str(slurm_task_id)+".json", 'w') as f_ann:  # write back to the JSON
    json.dump(train, f_ann, indent=2)


#f = open("/home/azstaszewska/Data/Detectron full set/final/batch_temp/val_"+str(slurm_task_id)+".json", "x")

with open("/home/azstaszewska/Data/Detectron full set/final/batch_temp/val_"+str(slurm_task_id)+".json", 'w') as f_ann:  # write back to the JSON
    json.dump(val, f_ann, indent=2)



for d in ["train", "val"]:
	DatasetCatalog.register("dataset_"+d, lambda d=d: json.load(open("/home/azstaszewska/Data/Detectron full set/final/batch_temp/"+d +"_"+str(slurm_task_id)+".json")))
	MetadataCatalog.get("dataset_"+d).set(thing_classes=['lack of fusion', 'keyhole'])


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

cfg.INPUT.MAX_SIZE_TRAIN = 2500

cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ("dataset_val",)

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
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
cfg.OUTPUT_DIR = "/home/azstaszewska/Batch models/model_"+str(slurm_task_id)
# make the output directory
os.makedirs(Path(cfg.OUTPUT_DIR), exist_ok=True)
os.makedirs(Path("/home/azstaszewska/Batch models/output_"+str(slurm_task_id)), exist_ok=True)

trainer = CocoTrainer(cfg)  # create trainer object from cfg
trainer.resume_or_load(resume=False)  # start training from iteration 0
trainer.train()  # train the model!

model_checkpoints = sorted(Path(cfg.OUTPUT_DIR).glob('*.pth'))  # paths to weights saved druing training
cfg.DATASETS.TEST = ("dataset_val",)  # predictor requires this field to not be empty
cfg.MODEL.WEIGHTS = str(model_checkpoints[-1])  # use the last model checkpoint saved during training. If you want to see the performance of other checkpoints you can select a different index from model_checkpoints.

predictor = DefaultPredictor(cfg)  # create predictor object
print(predictor)
evaluator = COCOEvaluator("dataset_val", output_dir="/home/azstaszewska/Batch models/output_"+str(slurm_task_id))
print(evaluator)

val_loader = build_detection_test_loader(cfg, 'dataset_val')
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
res = evaluator.evaluate()

results = open("/home/azstaszewska/batch_run_results_corrected.txt", "a")
results.write(str(dict(metrics))+",\n")
results.write(str(dict(res))+",\n")
results.close()



#train_loader = build_detection_test_loader(cfg, 'dataset_train')
#inference_on_dataset(trainer.model, train_loader, evaluator)
