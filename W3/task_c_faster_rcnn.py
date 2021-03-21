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
import getDicts

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


LOCAL_RUN = False
IMG_PATH = "../datasets/KITTI-MOTS/training/image_02"
LABEL_PATH = "../datasets/KITTI-MOTS/instances_txt"
BASE_PATH = "../datasets/KITTI-MOTS/"


"""
IMG_PATH = "../datasets/MOTSChallenge/train/images"
LABEL_PATH = "../datasets/MOTSChallenge/train/instances_txt"
BASE_PATH = "../datasets/MOTSChallenge/train/"
"""

if LOCAL_RUN:
    IMG_PATH = "../resources/KITTI-MOTS/training/image_02"
    LABEL_PATH = "../resources/KITTI-MOTS/instances_txt"
    BASE_PATH = "../resources/KITTI-MOTS/"


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)

dataset_dicts = getDicts.get_dicts(BASE_PATH,IMG_PATH)
train,val,test = getDicts.split_data(dataset_dicts)


DatasetCatalog.register("kitti-mots_train", lambda d="train": train)
MetadataCatalog.get("kitti-mots_train").set(thing_classes=["Car","Pedestrian"])
DatasetCatalog.register("kitti-mots_val", lambda d="val": val)
MetadataCatalog.get("kitti-mots_val").set(thing_classes=["Car", "Pedestrian"])
kitti_mots_metadata = MetadataCatalog.get("kitti-mots_val")


for rand in random.sample(val, 1):
        img = cv2.imread(rand["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(rand)
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
        cv2.imwrite("out_kittimots_faster_rcnn_task_c.png", out.get_image()[:, :, ::-1])


evaluator = COCOEvaluator("kitti-mots_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "kitti-mots_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

