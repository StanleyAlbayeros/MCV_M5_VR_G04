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
from detectron2.utils.visualizer import Visualizer
from pycocotools import coco

import getDicts
from src import LossEvalHook, config, graphicUtils, imgUtils

### alias etseM5="ssh group04@158.109.75.51 -p 55022"


def parse_args():
    parser = argparse.ArgumentParser(description="Task a")
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
        "-m",
        "--mots",
        dest="mots",
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
    model_group,
    geninference=False,
):
    current_output_dir = f"{config.output_path}/models/COCO_KITTI_MOTSC/{model_name}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    cfg.DATASETS.TEST = ("validation",)
    cfg.DATASETS.TRAIN = ("training_kitti", "training_motsc", )
    cfg.DATALOADER.NUM_WORKERS = 8

    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################

    # # Let training initialize from model zoo
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1500  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        256  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config.thing_classes)
    tasks = ("bbox", "segm")
    # cfg.INPUT.MIN_SIZE_TEST = 700
    # cfg.INPUT.MAX_SIZE_TEST = 600
    # cfg.INPUT.MAX_SIZE_TRAIN = 600
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    """ 
    # cfg.TEST.EVAL_PERIOD = 20
    # cfg.SOLVER.CHECKPOINT_PERIOD = 100


    # 
    # class MyTrainer(DefaultTrainer):
    #     @classmethod
    #     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #         if output_folder is None:
    #             output_folder = cfg.OUTPUT_DIR
    #         return COCOEvaluator(dataset_name, tasks, True, output_folder)
                        
    #     def build_hooks(self):
    #         hooks = super().build_hooks()
    #         hooks.insert(-1,LossEvalHook.LossEvalHook(
    #             cfg.TEST.EVAL_PERIOD,
    #             self.model,
    #             build_detection_test_loader(
    #                 self.cfg,
    #                 self.cfg.DATASETS.TEST[0],
    #                 DatasetMapper(self.cfg,True, )
    #             )
    #         ))
    #         return hooks 
    """

    start = time.time()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################

    # cfgdmp = str(cfg.dump())
    if geninference:
        if config.verbose:
            print(colorama.Fore.LIGHTMAGENTA_EX + "\tInference start")

        # build = build_model(cfg)
        # conda env export > detect2.yml
        evaluator = COCOEvaluator(
            "validation",
            tasks,
            use_fast_impl=False,
            output_dir=f"{current_output_dir}",
            distributed=True,
        )
        val_loader = build_detection_test_loader(cfg, "validation")

        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        end = time.time()
        time_elapsed = end - start

        print(f"{model_name} #RESULTS#")
        print(str(results) + f"\n{time_elapsed}")
        print(f"{model_name} #RESULTS#")

        txt_results_path = f"{config.txt_results_path}/COCO_KITTI_MOTSC"
        config.create_txt_results_path(txt_results_path)

        with open(f"{txt_results_path}/{model_name}.txt", "w") as writer:
            writer.write(str(results))
            writer.write(f"\n{time_elapsed}")
            if config.verbose:
                print(colorama.Fore.YELLOW + f"{results}")
        if config.verbose:
            print(colorama.Fore.LIGHTMAGENTA_EX + "\tInference end")
        
        results = str(results) + f"\n{time_elapsed}"

    # MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # exit()
    return results


if __name__ == "__main__":
    ##########################################################################################
    ###################################   WORKSPACE SETUP   ##################################
    ##########################################################################################
    colorama.init(autoreset=False)
    parser, fname = parse_args()
    local = parser.local
    v = parser.verbose
    geninference = parser.geninference
    mots = parser.mots

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
    ##############################   TRAIN MODELS WITH NEW DATA   ############################
    ##########################################################################################

    for model_name, model_url in config.mask_rcnn_models.items():
        if config.verbose:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        config.mask_rcnn_results[f"{model_name}"] = use_model(
            model_name=model_name,
            model_url=model_url,
            model_group="KITTI_MOTSC",
            geninference=geninference,
        )
        # break
    for model_name, result in config.mask_rcnn_results.items():
        print(f"{model_name}: {result}\n\n\n\n")

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
