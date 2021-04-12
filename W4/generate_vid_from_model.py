import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
import getDicts
from pycocotools import coco
import sys
import colorama
import random
from src import config
import logging
import time
from tqdm import tqdm

from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    DatasetMapper,
)
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from src import imgUtils

### 158.109.75.51 –p 55022


def parse_args():
    parser = argparse.ArgumentParser(description="visualize annotations")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--mots",
        dest="mots",
        action="store_true",
        default=False,
        required=False,
    )
    fname = os.path.splitext(parser.prog)
    return parser.parse_args(), fname


def use_model(model_name, model_url, training_dataset, validation_dataset, metadata):
    training_sets = "COCO_KITTI_MOTSC"
    # training_sets = "COCO_KITTI"
    current_output_dir = f"outputs/task_b/models/{training_sets}/{model_name}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"

    ## CHANGE BETWEEN CHECKPOINT AND TRAINED MODEL FOR TASK A_B_C
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.DATASETS.TEST = ("validation",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config.thing_classes)

    predictor = DefaultPredictor(cfg)
    i = 0
    for d in tqdm(validation_dataset, desc="Image_gen", colour="Magenta"):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("training_kitti"),
            scale=0.8,
            instance_mode=ColorMode.SEGMENTATION,
        )

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]
        # cv2.imshow("img", img)
        # cv2.waitKey(1)

        scale_percent = 60  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        tmpname = d["file_name"]
        tmpname_list = tmpname.split(os.sep)
        imgid = d["image_id"]
        filepath = f"{config.gen_img_path}/{training_sets}/{model_name}/PNG/{tmpname_list[-2]}"
        filename = f"{filepath}/{imgid}.png"
        config.create_txt_results_path(filepath)

        

        cv2.imwrite(filename, img)

    # for d in training_dataset:
    #     im = cv2.imread(d["file_name"])
    #     v = Visualizer(im[:, :, ::-1],
    #                 metadata=MetadataCatalog.get("KITTI_MOTS_train"),
    #                 scale=0.8,
    #                 # instance_mode=ColorMode.IMAGE_BW
    #                 instance_mode=ColorMode.IMAGE_BW
    #     )
    #     out = v.draw_dataset_dict(d)
    #     cv2.imshow("img", out.get_image()[:, :, ::-1])
    #     cv2.waitKey(1)


if __name__ == "__main__":
    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    colorama.init(autoreset=False)
    parser, fname = parse_args()
    mots = parser.mots

    v = parser.verbose
    if v:
        print(
            colorama.Fore.LIGHTRED_EX
            + "\n#################################\n"
            + str(parser)
        )

    config.init_workspace(v, fname[0])
    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##########################################################################################
    ###############################   DATASET + METADATA SETUP   #############################
    ##########################################################################################

    if config.verbose:
        print(colorama.Fore.LIGHTMAGENTA_EX + "\nGetting dataset train val split")

    train, val = getDicts.generate_datasets(mots)

    print(f"Train and val datasets generated")

    DatasetCatalog.register(
        "training_kitti",
        lambda: getDicts.register_helper(config.train_pkl_kitti_mots),
    )
    MetadataCatalog.get("training_kitti").set(thing_classes=config.thing_classes)
    MetadataCatalog.get("training_kitti").thing_colors = config.thing_colors
    DatasetCatalog.register(
        "training_motsc",
        lambda: getDicts.register_helper(config.train_pkl_mots_challenge),
    )
    MetadataCatalog.get("training_motsc").set(thing_classes=config.thing_classes)

    DatasetCatalog.register(
        "validation", lambda: getDicts.register_helper(config.validation_pkl)
    )
    MetadataCatalog.get("validation").set(thing_classes=config.thing_classes)

    if config.verbose:
        print(colorama.Fore.LIGHTMAGENTA_EX + "Done getting dataset train val split\n")
    ##########################################################################################
    ###############################   DATASET + METADATA SETUP   #############################
    ##########################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################

    KITTI_MOTS_metadata = MetadataCatalog.get("training")

    for model_name, model_url in config.mask_rcnn_models.items():
        if config.verbose:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        config.mask_rcnn_results[f"{model_name}"] = use_model(
            model_name=model_name,
            model_url=model_url,
            training_dataset=train,
            validation_dataset=val,
            metadata=KITTI_MOTS_metadata,
        )
        # break
    for model_name, result in config.mask_rcnn_results.items():
        if config.verbose:
            print(f"{model_name}: {result}\n\n\n\n")

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
