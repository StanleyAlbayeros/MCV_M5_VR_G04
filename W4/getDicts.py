import os
import pickle

import colorama
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools
import pycocotools.mask as rletools
from detectron2.structures import BoxMode
from PIL import Image
from tqdm import tqdm

from src import config
import io_tools

"""
Save catalogues to pkl
"""


def catalog_to_pkl(catalog, path):
    with open(path, "wb") as f:
        pickle.dump(catalog, f)
        f.close()


def pkl_to_catalog(path):
    catalog = []
    with open(path, "rb") as f:
        catalog = pickle.load(f)
        f.close()
    return catalog


"""
Split para train i val 
"""


def generate_motschallenge_pkls(db_path, img_path, train_pkl, val_pkl):
    train_dataset = {}

    for file in sorted(os.listdir(db_path + "/train/instances_txt")):
        annotations = io_tools.load_txt(db_path + "/train/instances_txt/" + file)
        train_samples = len(annotations)
        train = {}
        for key in range(1, train_samples):
            train[key] = annotations[key]

        train_dataset[f"{file[0:-4]}"] = train

    if os.path.exists(train_pkl):

        if config.verbose:
            print(colorama.Fore.BLUE + f"\t Found {train_pkl} pkl, loading")
        train_catalog = pkl_to_catalog(train_pkl)

    else:

        if config.verbose:
            print(colorama.Fore.BLUE + "\t Generating train annotations")
        train_catalog = get_dicts(train_dataset, img_path, ".jpg")

        if config.verbose:
            print(colorama.Fore.MAGENTA + f"\t\tSaving {train_pkl} pkl")
        catalog_to_pkl(train_catalog, train_pkl)

        del train_catalog


def generate_kitti_mots_pkls(
    base_path,
    images_path,
    train_pkl,
    val_pkl,
    extension=".png",
    random_train_test=False,
):
    validation = [2, 6, 7, 8, 10, 13, 14, 16, 18]
    validation = np.char.zfill(list(map(str, validation)), 4)

    # training = ["2","6","7","8","10","13","14","16","18"]
    # training = np.char.zfill(training, 4)

    if random_train_test:
        intlist = random.sample(range(0, 20), 9)
        validation = np.char.zfill(list(map(str, intlist)), 4)
    raw_dicts = []
    train_dataset = {}
    val_dataset = {}

    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        if file[0:-4] not in validation:
            train_dataset[f"{file[0:-4]}"] = annotations
        else:
            val_dataset[f"{file[0:-4]}"] = annotations

    if os.path.exists(train_pkl):
        if config.verbose:
            print(colorama.Fore.BLUE + f"\tFound {train_pkl} pkl, loading")
        train_catalog = pkl_to_catalog(train_pkl)
    else:
        if config.verbose:
            print(
                colorama.Fore.BLUE
                + "\tGenerating kitti mots train for : 2,6,7,8,10,13,14,16,18 at: \n"
                + f"\t\t\t {train_pkl}"
            )
        train_catalog = get_dicts(train_dataset, images_path, extension)
        if config.verbose:
            print(colorama.Fore.MAGENTA + f"\t\tSaving {train_pkl} pkl")
        catalog_to_pkl(train_catalog, train_pkl)
        del train_catalog

    if os.path.exists(val_pkl):
        if config.verbose:
            print(colorama.Fore.BLUE + f"\tFound {val_pkl} pkl, loading")
        val_catalog = pkl_to_catalog(val_pkl)
    else:
        if config.verbose:
            print(
                colorama.Fore.BLUE
                + "\tGenerating kitti mots val annotations"
                + f"\t\t\t {val_pkl}"
            )
        val_catalog = get_dicts(val_dataset, images_path, extension)
        if config.verbose:
            print(colorama.Fore.MAGENTA + f"\t\tSaving {val_pkl} pkl")
        catalog_to_pkl(val_catalog, val_pkl)
        del val_catalog


def generate_combined_pkl(path):
    for file in sorted(os.listdir(path)):
        print(file)
    print(config.pkl_path)
    exit()


def register_helper(path):
    if os.path.exists(path):
        if config.verbose:
            print(colorama.Fore.CYAN + f"Retreiving information from {path}")
        catalog = pkl_to_catalog(path)
    return catalog


"""
Obtención de las boxes de cada imagen (KITTI-MOTS)
"""


def get_dicts(dataset, images_path, extension):
    raw_dicts = []
    dataset_dicts = []
    Pedestrians = []
    Cars = []
    folder_id = []

    for folder, annos in tqdm(dataset.items(), desc="Folder loop", colour="Cyan"):
        # for idx, dir in tqdm(enumerate(images_path), desc = "Folder loop", colour="Cyan"):
        # print(folder)
        # exit()
        for key, anno in tqdm(annos.items(), desc="Annons loop", colour="Magenta"):
            record = {}
            img_id = str(key).zfill(6)
            img_path = os.path.join(images_path, folder, str(img_id) + extension)
            img = cv2.imread(img_path)
            height, width, channels = img.shape

            record["file_name"] = img_path
            record["image_id"] = img_id
            record["height"] = height
            record["width"] = width
            objs = []

            for instance in anno:

                category_id = instance.class_id

                # if category_id == 1 or category_id == 2:
                bbox = pycocotools.mask.toBbox(instance.mask)
                mask = rletools.decode(instance.mask)
                contours, _ = cv2.findContours(
                    (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                segmentation = [[int(i) for i in c.flatten()] for c in contours]
                segmentation = [s for s in segmentation if len(s) >= 6]

                if not segmentation:
                    continue

                # if len(segmentation) == 0:
                #     continue
                # End: convert rle to poly
                # print (segmentation)

                # thing_classes = ["Person", "Other", "Car"]
                if category_id == 1 or category_id == 2:
                    obj = {
                        "bbox": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        # "type": 'Car' if category_id==2 else 'Person',
                        "category_id": 2 if category_id == 1 else 0,
                        "segmentation": segmentation,
                        # "isCrowd": 0,
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

            # print(record)
            # exit()

            # break
        # break
    return dataset_dicts
