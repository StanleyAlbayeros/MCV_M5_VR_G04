import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import io_tools
import pycocotools
from tqdm import tqdm
from detectron2.structures import BoxMode
import pycocotools.mask as rletools
import pickle

"""
Split para train i val
"""
def split_data_kitti_motts(base_path):
    training = ["2","6","7","8","10","13","14","16","18"]
    training = np.char.zfill(training, 4)
    raw_dicts = []
    train_dataset = {}
    val_dataset = {}
    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        if file[0:-4] in training:
            train_dataset[f"{file[0:-4]}"] = annotations
        else:
            val_dataset[f"{file[0:-4]}"] =  annotations

    return train_dataset,val_dataset


"""
Obtención de las boxes de cada imagen (KITTI-MOTS)
"""
def get_dicts(dataset,images_path,masks_path,extension=".png"):
    raw_dicts = []
    dataset_dicts = []
    Pedestrians = []
    Cars =[]
    folder_id = []
    """for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        raw_dicts.append(annotations)
        folder_id.append(file[:-4])"""
    for folder,annos in dataset.items():
        for anno in annos:
            record = {}
            img_id = str(anno).zfill(6)
            img_path = os.path.join(images_path,folder,str(img_id)+extension)
            print(anno)
            img = cv2.imread(img_path)
            height,width,channels = img.shape

            record["file_name"] = img_path
            record["image_id"] = img_id
            record["height"] = height
            record["width"] = width
            objs = []

            for instance in anno.items():

                category_id = instance.class_id

                if category_id == 1 or category_id == 2:
                    bbox = pycocotools.mask.toBbox(instance.mask)
                    mask = rletools.decode(instance.mask)
                    segmentation = []
                    contours, _ = cv2.findContours((mask).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        contour = contour.flatten().tolist()
                        # segmentation.append(contour)
                        if len(contour) > 4:
                            segmentation.append(contour)
                    if len(segmentation) == 0:
                        continue
                        # End: convert rle to poly
                        # print (segmentation)

                    """ori_class = int(instance[3])
                    if ori_class == 1:
                        transform_class = 0
                    elif ori_class == 2:
                        transform_class = 1
                    else:
                        # transform_class = 2
                        continue"""
                    obj = {
                        "bbox": [float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": category_id -1,
                        "segmentation" : segmentation
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            break
        break
    return dataset_dicts


def getMask(mask_path,img_path):
    masks = io_tools.load_image(os.path.join(mask_path,"0001/000005.png"))
    return masks