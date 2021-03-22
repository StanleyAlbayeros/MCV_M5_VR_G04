"""

Task 2: Use object detection models in inference: Faster R-CNN

"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

from detectron2.utils.visualizer import ColorMode

from PIL import Image
import io_tools
import pycocotools
from tqdm import tqdm
import getDicts
import torch

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

LOCAL_RUN = True
"""
IMG_PATH = "../datasets/KITTI-MOTS/training/image_02"
LABEL_PATH = "../datasets/KITTI-MOTS/instances_txt"
BASE_PATH = "../datasets/KITTI-MOTS/"
"""


IMG_PATH = "../datasets/MOTSChallenge/train/images"
LABEL_PATH = "../datasets/MOTSChallenge/train/instances_txt"
BASE_PATH = "../datasets/MOTSChallenge/train/"



if LOCAL_RUN:
    IMG_PATH = "../resources/MOTSChallenge/train/images"
    LABEL_PATH = "../resources/MOTSChallenge/instances_txt"
    BASE_PATH = "../resources/MOTSChallenge/train/"



dataset_dicts = getDicts.get_dicts(BASE_PATH,IMG_PATH,".jpg")
train,val,test = getDicts.split_data(dataset_dicts)


DatasetCatalog.register("kitti-mots_train", lambda d="train": train)
MetadataCatalog.get("kitti-mots_train").set(thing_classes=["Car","Pedestrian"])
DatasetCatalog.register("kitti-mots_val", lambda d="val": val)
MetadataCatalog.get("kitti-mots_val").set(thing_classes=["Car", "Pedestrian"])
kitti_mots_metadata = MetadataCatalog.get("kitti-mots_val")

kitti_mots_metadata = MetadataCatalog.get("kitti-mots_train")

cfg = get_cfg()
MODEL_IN_USE = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(MODEL_IN_USE))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_IN_USE)
cfg.DATASETS.TRAIN = ("kitti-mots_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.BASE_LR = 0.001  
cfg.SOLVER.STEPS = []
cfg.SOLVER.MAX_ITER = 300    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


torch.cuda.empty_cache()
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


# for rand_image in random.sample(val, 1):
#     img = cv2.imread(rand_image["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(rand_image)
#     plt.imshow(out.get_image()[:, :, ::-1])
#     plt.show()
#     cv2.imwrite("out_motschallenge_faster_rcnn_task_d.png", out.get_image()[:, :, ::-1])

torch.cuda.empty_cache()
evaluator = COCOEvaluator("kitti-mots_val", cfg, False, output_dir="./outputs/")
val_loader = build_detection_test_loader(cfg, "kitti-mots_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))