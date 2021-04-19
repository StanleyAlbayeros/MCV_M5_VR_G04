import argparse
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

import colorama
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools import coco
from tqdm import tqdm

import getDicts
from src import config, imgUtils

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


def use_model(model_name, model_url):
    current_output_dir = f"{config.output_path}/{model_name}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)

    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    # just swap the comments on the next 4 lines to generate either ooc or coco samples
    # out_of_contex_dir = "../resources/out_of_context"
    mode = "coco_mod"
    # out_of_contex_dir = "datasets/coco_mod/mods"
    # out_of_contex_dir = "datasets/coco_mod_c/mods"
    # out_of_contex_dir = "datasets/coco/val2017"
    out_of_contex_dir = "datasets/coco_mod_d/mods"

    predictor = DefaultPredictor(cfg)

    images_in_dir = os.listdir(out_of_contex_dir)

    # ooc only has 43 images, so check for that anyways. Only take a random sample if 
    # it's the coco val dataset because qualitative results in 5k images is dumb
    # if len(images_in_dir) > 43:
    #     mode = "coco"
    #     list_of_images = random.sample(images_in_dir, 25)



    for filename in tqdm(images_in_dir, desc="Image_gen", colour="Magenta"):
        
        img_no_ext = Path(f"{filename}").stem

        im = cv2.imread(os.path.join(out_of_contex_dir, filename))
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get("coco_2017_val"),
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,
        )
        ##### to draw all classes, pass instances to line 98, or "dogs" and change class
        ##### index to pass a single class
        instances = outputs["instances"].to("cpu")
        # print(MetadataCatalog.get("coco_2017_val"))
        dogs = instances[instances.pred_classes == 77]
        # print(instances)
        v = v.draw_instance_predictions(instances)
        img = v.get_image()[:, :, ::-1]
        # cv2.imshow("img", img)

        # k = cv2.waitKey(0)
        # if k==27:    # Esc key to stop
        #     exit()

        filepath_originals = f"{config.gen_img_path}/task_d_outputs/originals"
        filename_original = f"{filepath_originals}/{img_no_ext}_original.png"
        filepath_out = f"{config.gen_img_path}/task_d_outputs/result"
        filename_out = f"{filepath_out}/{img_no_ext}_out.png"

        config.create_txt_results_path(filepath_originals)
        config.create_txt_results_path(filepath_out)
        
        ###uncomment below only if you also want the originals
        cv2.imwrite(filename_original, im)
        cv2.imwrite(filename_out, img)


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

    config.init_workspace(parser, fname[0])
    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################

    for model_name, model_url in config.mask_rcnn_models.items():
        if config.verbose:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        config.mask_rcnn_results[f"{model_name}"] = use_model(
            model_name=model_name, model_url=model_url
        )
        # break
    for model_name, result in config.mask_rcnn_results.items():
        if config.verbose:
            print(f"{model_name}: {result}\n\n\n\n")

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
