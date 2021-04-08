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

def split_data_mots_challenge(db_path,img_path,train_pkl, val_pkl, v= True):
    train_dataset = {}
    val_dataset = {}
    for file in sorted(os.listdir(db_path + "/train/instances_txt")):
        annotations = io_tools.load_txt(db_path+"/train/instances_txt/"+file)
        train_samples = int(np.floor(len(annotations) * 0.6))
        val_samples = int(np.floor(len(annotations) * 0.4))
        train = {}
        val = {}
        for key in range(1,train_samples):
            train[key] = annotations[key]
        for key in range(train_samples,val_samples):
            val[key] = annotations[key]
        train_dataset[f"{file[0:-4]}"] = train
        val_dataset[f"{file[0:-4]}"] = val
    if os.path.exists(train_pkl):
        if v: print(colorama.Fore.BLUE + "\t Found train pkl, loading")
        train_catalog = pkl_to_catalog(train_pkl)
    else:
        if v: print(colorama.Fore.BLUE + "\t Generating train annotations")
        train_catalog = get_dicts(train_dataset, img_path, ".jpg")
        if v: print(colorama.Fore.MAGENTA + "\t\tSaving train pkl")
        catalog_to_pkl(train_catalog,train_pkl)
        del train_catalog
    if os.path.exists(val_pkl):
        if v: print(colorama.Fore.BLUE + "\t Found val pkl, loading")
        val_catalog = pkl_to_catalog(val_pkl)
    else:
        if v: print(colorama.Fore.BLUE + "\t Generating val annotations")
        val_catalog = get_dicts(val_dataset, img_path, ".jpg")
        if v: print(colorama.Fore.MAGENTA + "\t\tSaving val pkl")
        catalog_to_pkl(val_catalog,val_pkl)
        del val_catalog



def split_data_kitti_mots(
    base_path, images_path, train_pkl, val_pkl,v =False, extension=".png", random_train_test=False
):
    # training = [2,6,7,8,10,13,14,16,18]
    # training = np.char.zfill(list(map(str, training)), 5)

    training = ["2","6","7","8","10","13","14","16","18"]
    training = np.char.zfill(training, 4)

    if random_train_test:
        intlist = random.sample(range(0, 20), 9)
        training = np.char.zfill(list(map(str, intlist)), 5)
    raw_dicts = []
    train_dataset = {}
    val_dataset = {}

    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        if file[0:-4] in training:
            train_dataset[f"{file[0:-4]}"] = annotations
        else:
            val_dataset[f"{file[0:-4]}"] = annotations

    if os.path.exists(train_pkl):
        if v: print(colorama.Fore.BLUE + "\tFound train pkl, loading")
        train_catalog = pkl_to_catalog(train_pkl)
    else:
        if v: print(colorama.Fore.BLUE + "\tGenerating train annotations")
        train_catalog = get_dicts(train_dataset, images_path, extension)
        if v: print(colorama.Fore.MAGENTA + "\t\tSaving train pkl")
        catalog_to_pkl(train_catalog, train_pkl)
        del train_catalog

    if os.path.exists(val_pkl):
        if v: print(colorama.Fore.BLUE + "\tFound val pkl, loading")
        val_catalog = pkl_to_catalog(val_pkl)
    else:
        if v: print(colorama.Fore.BLUE + "\tGenerating val annotations")
        val_catalog = get_dicts(val_dataset, images_path, extension)
        if v: print(colorama.Fore.MAGENTA + "\t\tSaving val pkl")
        catalog_to_pkl(val_catalog, val_pkl)
        del val_catalog



def register_helper(path, v):
    if os.path.exists(path):
        if v: print(colorama.Fore.CYAN + f"Retreiving information from {path}")
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


                if category_id == 1 or category_id == 2:
                    bbox = pycocotools.mask.toBbox(instance.mask)
                    mask = rletools.decode(instance.mask)
                    segmentation = []
                    contours, _ = cv2.findContours(
                        (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        contour = contour.flatten().tolist()
                        # segmentation.append(contour)
                        if len(contour) > 4:
                            segmentation.append(contour)
                    if len(segmentation) == 0:
                        continue
                        # End: convert rle to poly
                        # print (segmentation)

                    obj = {
                        "bbox": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        ],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": 0 if category_id==1 else 1,
                        "type": 'Car' if category_id==1 else 'Pedestrian',
                        "segmentation": segmentation,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            #break
        #break
    return dataset_dicts
