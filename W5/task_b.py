import argparse
import logging
import os
import pickle
import random
import sys
import time

import colorama
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pycocotools.mask as rletools
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools import coco
from tqdm import tqdm

import getDicts
from src import config, imgUtils, cooc_tools

### 158.109.75.51 –p 55022


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task_b: get co-occurrence matrices on coco_2017_val / generate images for coco_mod set"
    )
    fname = os.path.splitext(parser.prog)
    parser.add_argument(
        "-v",
        "--verbose",
        help="Display more information and progress bars on execution",
        dest="verbose",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-r",
        "--run_model",
        help="Run the model for image or data generation",
        dest="run_model",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-g",
        "--generate_images",
        help="Generate classified images",
        dest="generate_images",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        help=f"Default = results.csv. Filename for the CSV, saved in {fname[0]}/csv/",
        required=False,
        default="results.csv",
    )
    parser.add_argument(
        "-plt",
        "--plt_filename",
        help="Plot filename",
        default="cooc_heatmap.png",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--load_csv",
        help="Load csv from path",
        dest="load_csv",
        action="store_true",
        default=False,
        required=False,
    )
    if parser.parse_args().verbose:
        print(
            colorama.Fore.LIGHTRED_EX
            + "\n#################################\n"
            + str(parser.parse_args())
        )
    return parser.parse_args(), fname


def show_img_with_escape(img):
    cv2.imshow("img", img)
    k = cv2.waitKey(0)
    if k == 27:  # Esc key to stop
        exit()


def use_model(model_name, model_url):
    current_output_dir = f"{config.output_path}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)

    coco_2017_val_dataset = "coco_2017_val"
    cfg.DATASETS.TEST = (coco_2017_val_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    # out_of_contex_dir = "../resources/out_of_context"
    out_of_contex_dir = "datasets/coco/val2017"

    predictor = DefaultPredictor(cfg)
    i = 0
    pred_dict = {}
    pred_dict["Class_names"] = MetadataCatalog.get(coco_2017_val_dataset).thing_classes

    for filename in tqdm(
        os.listdir(out_of_contex_dir), desc="Image_gen", colour="Magenta"
    ):

        im = cv2.imread(os.path.join(out_of_contex_dir, filename))

        outputs = predictor(im)
        # v = Visualizer(
        #     im[:, :, ::-1],
        #     metadata=MetadataCatalog.get(coco_2017_val_dataset),
        #     scale=1,
        #     instance_mode=ColorMode.IMAGE_BW,
        # )

        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # img = v.get_image()[:, :, ::-1]

        # show_img_with_escape(img)

        img_filename = f"{config.gen_img_path}/{filename}"
        # print(filename)

        # cv2.imwrite(img_filename, img)

        ######################## Saving class data for cooc matrix  ############################
        ############################# text class idx to labels #################################
        pred_classes = outputs["instances"].pred_classes
        class_names = MetadataCatalog.get(coco_2017_val_dataset).thing_classes
        pred_class_names = list(map(lambda x: class_names[x], pred_classes))
        # print(pred_classes)
        # print(f"pred class names {pred_class_names} \n")
        # print(class_names)

        tmparr = [0] * len(class_names)

        # print(pred_dict)
        # exit()

        for idx in pred_classes:
            tmparr[idx] += 1

        #### Save data to pred_dict dictionary which is converted to a co-occurrence matrix ####
        pred_dict[f"{filename[:-4]}"] = tmparr

    # print(cooc)
    #############################################################################################
    ####################### Build co-occurrence matrix from dict ################################
    ############################### And save to image ###########################################

    cooc_mat = cooc_tools.build_cooc_matrix(pred_dict)
    return cooc_mat


def run_model():
    for model_name, model_url in config.mask_rcnn_models.items():
        if config.verbose:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        return use_model(model_name=model_name, model_url=model_url)


if __name__ == "__main__":
    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    colorama.init(autoreset=False)

    parser, fname = parse_args()

    config.init_workspace(parser, fname[0])

    if config.verbose:
        logging.basicConfig(level=logging.INFO)

    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################

    cooc_mat = []
    if config.run_model is True:
        cooc_mat = run_model()
        cooc_tools.save_dataframe(cooc_mat, "cooc_df")
        cooc_tools.save_cooc_plot(cooc_mat)
    # else:

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
