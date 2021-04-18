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


def coco_retrain(model_name, model_url):
    current_output_dir = f"{config.output_path}/coco_retrain/"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_url))
    os.makedirs(current_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = f"{current_output_dir}"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
    cfg.DATASETS.TEST = ("coco_2017_val",)

    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1500  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        256  # faster, and good enough for this toy dataset (default: 512)
    )
    tasks = ("segm", )

    start = time.time()
    predictor = DefaultPredictor(cfg)

    if config.verbose:
        print(colorama.Fore.LIGHTMAGENTA_EX + "\tInference start")

    evaluator = COCOEvaluator(
        "validation",
        tasks,
        use_fast_impl=False,
        output_dir=f"{current_output_dir}",
        distributed=True,
    )
    val_loader = build_detection_test_loader(cfg, "coco_2017_val")

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
