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

### ssh group04@158.109.75.51 -p 55022


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
    model_group,
    training_dataset,
    validation_dataset,
    metadata,
    geninference=False,
    generate_img=False,
):
    current_output_dir = f"{config.output_path}/models/COCO_KITTI/{model_name}"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
    cfg.DATASETS.TRAIN = ("KITTI_MOTS_training",)
    cfg.DATALOADER.NUM_WORKERS = 8

    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################

    # # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        256  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config.thing_classes)

    start = time.time()
    # trainer = MyTrainer(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if config.verbose:
        print(f"\n\n\nUsing config: {cfg.dump()}\n\n\n")

    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################

    cfgdmp = str(cfg.dump())
    if geninference:
        if config.verbose:
            print(colorama.Fore.LIGHTMAGENTA_EX + "\tInference start")

        # build = build_model(cfg)
        # conda env export > detect2.yml
        evaluator = COCOEvaluator(
            "KITTI_MOTS_val",
            tasks,
            use_fast_impl=False,
            output_dir=f"{current_output_dir}",
            distributed=True,
        )
        val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")

        results = inference_on_dataset(predictor.model, val_loader, evaluator)

        print(f"{model_name} #RESULTS#")
        print(results)
        print(f"{model_name} #RESULTS#")

        end = time.time()
        time_elapsed = end - start
        txt_results_path = f"{config.output_path}/txt_results/{model_group}"
        config.create_txt_results_path(txt_results_path)

        with open(f"{txt_results_path}/{model_name}.txt", "w") as writer:
            writer.write(str(results))
            writer.write(f"\n\n{cfgdmp}")
            if config.verbose:
                print(colorama.Fore.YELLOW + f"{results}")
        if config.verbose:
            print(colorama.Fore.LIGHTMAGENTA_EX + "\tInference end")
    # MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    if generate_img:
        imgUtils.generate_sample_imgs(
            target_metadata=metadata.get(cfg.DATASETS.TEST[0]),
            target_dataset=validation_dataset,
            output_path=config.output_path,
            add_str="_val",
            predictor=predictor,
            scale=1,
            num_imgs=10,
            model_name=model_name,
        )
        imgUtils.generate_sample_imgs(
            target_metadata=metadata.get(cfg.DATASETS.TRAIN[0]),
            target_dataset=training_dataset,
            output_path=config.output_path,
            add_str="_train",
            predictor=predictor,
            scale=1,
            num_imgs=10,
            model_name=model_name,
        )
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
    generate_samples = parser.generate_samples

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

    getDicts.generate_kitti_mots_pkls(
        config.db_path_kitti_mots,
        config.imgs_path_kitti_mots,
        config.train_pkl_kitti_mots,
        config.val_pkl_kitti_mots,
    )
    print(f"Train and val datasets generated")

    DatasetCatalog.register(
        "KITTI_MOTS_training",
        lambda: getDicts.register_helper(config.train_pkl_kitti_mots),
    )
    MetadataCatalog.get("KITTI_MOTS_training").set(thing_classes=config.thing_classes)

    DatasetCatalog.register(
        "KITTI_MOTS_val", lambda: getDicts.register_helper(config.val_pkl_kitti_mots)
    )
    MetadataCatalog.get("KITTI_MOTS_val").set(thing_classes=config.thing_classes)
    train = getDicts.register_helper(config.train_pkl_kitti_mots)
    val = getDicts.register_helper(config.val_pkl_kitti_mots)

    if config.verbose:
        print(colorama.Fore.LIGHTMAGENTA_EX + "Done getting dataset train val split\n")
    ##########################################################################################
    ###############################   DATASET + METADATA SETUP   #############################
    ##########################################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################

    KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_training")

    for model_name, model_url in config.mask_rcnn_models.items():
        if config.verbose:
            print(
                colorama.Fore.LIGHTGREEN_EX
                + f"\nUsing {model_name} from url {model_url}"
            )
        config.mask_rcnn_results[f"{model_name}"] = use_model(
            model_name=model_name,
            model_url=model_url,
            model_group="COCO",
            training_dataset=train,
            validation_dataset=val,
            metadata=KITTI_MOTS_metadata,
            geninference=geninference,
            generate_img=generate_samples,
        )
        # break
    for model_name, result in config.mask_rcnn_results.items():
        print(f"{model_name}: {result}\n\n\n\n")

    ##########################################################################################
    ##############################   PRETRAINED MODEL INFERENCE   ############################
    ##########################################################################################
