import os
import io_tools
import pycocotools
from tqdm import tqdm
import cv2
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt
import numpy as np

"""
Obtención de las boxes de cada imagen (KITTI-MOTS)
"""
def get_dicts(base_path,images_path,extension=".png"):
    raw_dicts = []
    dataset_dicts = []
    Pedestrians = []
    Cars =[]
    folder_id = []
    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        raw_dicts.append(annotations)
        folder_id.append(file[:-4])
    for idx, dicts in tqdm(enumerate(raw_dicts)):
        for key,value in dicts.items():
            record = {}
            img_id = str(key).zfill(6)
            img_path = os.path.join(images_path,folder_id[idx].zfill(4),str(img_id)+extension)
            # print(img_path)
            img = cv2.imread(img_path)
            height,width,channels = img.shape

            record["file_name"] = img_path
            record["image_id"] = img_id
            record["height"] = height
            record["width"] = width
            objs = []
            for instance in value:
                category_id = instance.class_id
                if category_id == 1 or category_id == 2:
                    bbox = pycocotools.mask.toBbox(instance.mask)
                    obj = {
                        "bbox": [float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": category_id -1,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts


def split_data(dicts):
    tr_s = int(np.floor(len(dicts) * 0.6))
    val_s = int(np.floor(len(dicts) * 0.8))
    train_ds = dicts[:tr_s]
    val_ds = dicts[tr_s + 1: val_s]
    test_ds = dicts[val_s + 1:]

    return train_ds, val_ds, test_ds

