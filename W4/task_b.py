import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
import getDicts
import colorama
import random
import config

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

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
    return parser.parse_args()


if __name__ == "__main__":
    colorama.init(autoreset=False)
    parser = parse_args()
    local = parser.local
    v = parser.verbose
    generate_samples = parser.generate_samples
    if v:
        print(
            colorama.Fore.LIGHTRED_EX
            + "\n#################################\n"
            + str(parser)
        )

    config.init_workspace(local, v)
    if v:
        print(colorama.Fore.LIGHTMAGENTA_EX + "\nGetting dataset train val split")
    getDicts.split_data_kitti_motts(
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

    dtst = getDicts.register_helper(config.train_pkl, v)
    KITTI_MOTS_metadata = MetadataCatalog.get("KITTI_MOTS_training")
    if generate_samples: utils.generate_sample_imgs(KITTI_MOTS_metadata, dtst, v, config.output_path)

    #  config.py vars
    #  db_path
    #  masks_path
    #  imgs_path
    #  output_path
    #  pkl_path
    #  train_pkl
    #  val_pkl
    #  thing_classes