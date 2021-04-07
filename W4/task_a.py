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
import config

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from src import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Task b")
    parser.add_argument(
        "-l",
        "--local",
        dest="local",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-gs",
        "--generate_samples",
        dest="generate_samples",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "-i",
        "--geninference",
        dest="geninference",
        action="store_true",
        default=False,
        required=False,
    )
    fname = os.path.splitext(parser.prog)
    return parser.parse_args(), fname


def use_model(
    model_name,
    model_url,
    training_dataset,
    validation_dataset,
    metadata,
    v,
    geninference = False,
    print_samples = True,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    cfg.DATASETS.TRAIN = ("KITTI_MOTS_training",)
    cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    current_output_dir = f"{config.output_path}/models/{model_name}"
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    predictor = DefaultPredictor(cfg)


    utils.generate_sample_imgs(
        target_metadata=metadata,
        validation_dataset=validation_dataset,
        v=v,
        output_path=config.output_path,
        predictor=predictor,
        scale=0.5,
        num_imgs=10,
        model_name=model_name,
    )
    if geninference:
        build = build_model(cfg)
#conda env export > detect2.yml
    evaluator = COCOEvaluator(
        "KITTI_MOTS_val", cfg, False, output_dir=f"{current_output_dir}"
    )
    val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    txt_results_path = f'outputs/task_a/txt_results'
    os.makedirs(txt_results_path, exist_ok=True)
    #with open(f'{txt_results_path}/{model_name}.txt', 'w') as writer:
    #    writer.write(results)
    #    if v: print(colorama.Fore.YELLOW + f"{results}")
    print(f"{model_name} #RESULTS#")
    print(results)    
    print(f"{model_name} #RESULTS#")
	



if __name__ == "__main__":
    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    colorama.init(autoreset=False)
    parser, fname = parse_args()
    local = parser.local
    v = parser.verbose
    geninference = parser.geninference
    generate_samples = parser.generate_samples

    if v:
        print(
            colorama.Fore.LIGHTRED_EX
            + "\n#################################\n"
            + str(parser)
        )

    config.init_workspace(local, v, fname[0])

    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##########################################################################################
    ###############################   DATASET + METADATA SETUP   #############################
    ##########################################################################################

    if v:
        print(colorama.Fore.LIGHTMAGENTA_EX + "\nGetting dataset train val split")
    getDicts.split_data_kitti_mots(
        config.db_path, config.imgs_path, config.train_pkl, config.val_pkl, v
    )
    print(f"Train and val datasets generated")

    DatasetCatalog.register(
        "KITTI_MOTS_training", lambda: getDicts.register_helper(config.train_pkl, v)
    )
    MetadataCatalog.get("KITTI_MOTS_training").set(thing_classes=config.thing_classes)

    DatasetCatalog.register(
        "KITTI_MOTS_val", lambda: getDicts.register_helper(config.val_pkl, v)
    )
    MetadataCatalog.get("KITTI_MOTS_val").set(thing_classes=config.thing_classes)

    train = getDicts.register_helper(config.train_pkl, v)
    val = getDicts.register_helper(config.val_pkl, v)

    ##########################################################################################
    ###############################   DATASET + METADATA SETUP   #############################
    ##########################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################

    KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_training")

    for model_name, model_url in config.mask_rcnn_models.items():
        if v:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        use_model(
            model_name=model_name,
            model_url=model_url,
            training_dataset=train,
            validation_dataset=val,
            metadata=KITTI_MOTS_metadata,
            v=v,
        )

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
