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

LOCAL_RUN = False
DATA_TEST_PATH = "../datasets/KITTI-MOTS/testing/image_02"

if LOCAL_RUN:
    DATA_TEST_PATH = "../resources/KITTI-MOTS/testing/image_02"


def getIoU(predicted_bbox,gt_bbox):
    


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

for category_folder in os.listdir(DATA_TEST_PATH):
    for filename in random.sample(os.listdir(DATA_TEST_PATH + "/" + category_folder),3):
        print("validation image: "+DATA_TEST_PATH + "/" + category_folder + "/" + filename)
        im = cv2.imread(DATA_TEST_PATH+"/"+category_folder+"/"+filename)
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
