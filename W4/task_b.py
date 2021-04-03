import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as rletools

import getDicts


def init_path_vars(local_run=True):
    global db_path
    global masks_path
    global imgs_path
    global output_path
    global pkl_path
    global train_pkl
    global val_pkl

    output_path = "outputs/task_b"
    pkl_path = "datasetpkl"
    train_pkl = pkl_path + "/train.pkl"
    val_pkl = pkl_path + "/val.pkl"

    if local_run:
        db_path = "../resources/KITTI-MOTS"
        masks_path = "../resources/KITTI-MOTS/instances"
        imgs_path = "../resources/KITTI-MOTS/training/image_02"
    else:
        db_path = "../datasets/KITTI-MOTS"
        masks_path = "../datasets/KITTI-MOTS/instances"
        imgs_path = "../datasets/KITTI-MOTS/training/image_02"

    if not os.path.exists(output_path):
        print("Creating output dir")
        os.makedirs(output_path)
    if not os.path.exists(pkl_path):
        print("Creating pkl dir")
        os.makedirs(pkl_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Task b")
    parser.add_argument("--local", dest="local", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    local = parser.local
    init_path_vars(local)

    
    train, val = getDicts.split_data_kitti_motts(db_path, imgs_path, train_pkl, val_pkl)

    print(len(train))
    print(len(val))
