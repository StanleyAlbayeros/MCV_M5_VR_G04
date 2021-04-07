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
    parser.add_argument(
        "-km",
        "--kitti-Mots",
        dest = "kitti_mots",
        required = False,
        default= False,
        action="store_true",
        help = "Add kitti-mots to training dataset"
    )
    parser.add_argument(
        "-mc",
        "--MOTSChallenge",
        dest = "mots_challenge",
        required = False,
        action = "store_true",
        help = "Add MOTSChallenge to training dataset"
    )
    fname = os.path.splitext(parser.prog)
    return parser.parse_args(),fname


if __name__ == "__main__":
    colorama.init(autoreset=False)
    parser,fname = parse_args()
    kitti_mots = parser.kitti_mots
    mots_challenge = parser.mots_challenge

    local = parser.local
    v = parser.verbose

    generate_samples = parser.generate_samples
    if v:
        print(
            colorama.Fore.LIGHTRED_EX
            + "\n#################################\n"
            + str(parser)
        )
    local = False
    config.init_workspace(local, v,fname[0])
    print(config.train_pkl_kitti_mots)
    train_pkl = []
    val_pkl = []
    if v:
        print(colorama.Fore.LIGHTMAGENTA_EX + "\nGetting dataset train val split")
    if kitti_mots:
        getDicts.split_data_kitti_mots(
            config.db_path_kitti_mots, config.imgs_path_kitti_mots,
            config.train_pkl_kitti_mots, config.val_pkl_kitti_mots
        )

        train_pkl.append(config.train_pkl_kitti_mots)
        val_pkl.append(config.val_pkl_kitti_mots)

    if mots_challenge:
        getDicts.split_data_mots_challenge(
            config.db_path_mots_challenge, config.imgs_path_mots_challenge,
            config.train_pkl_mots_challenge, config.val_pkl_mots_challenge
        )
        train_pkl.append(config.train_pkl_mots_challenge)
        val_pkl.append(config.val_pkl_mots_challenge)
    print(f"Train and val datasets generated")

    DatasetCatalog.register(
        "training_set", lambda: getDicts.register_helper(train_pkl, v)
    )
    MetadataCatalog.get("training_set").set(thing_classes=config.thing_classes)
    DatasetCatalog.register(
        "val_set", lambda: getDicts.register_helper(val_pkl, v)
    )
    MetadataCatalog.get("KITTI_MOTS_val").set(thing_classes=config.thing_classes)

    dtst = getDicts.register_helper(train_pkl, v)

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