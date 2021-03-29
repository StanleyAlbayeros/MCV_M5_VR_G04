import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import io_tools
import pycocotools
from tqdm import tqdm
from detectron2.structures import BoxMode


"""
Obtención de las boxes de cada imagen (KITTI-MOTS)
"""
def get_dicts(base_path,images_path,masks_path,extension=".png"):
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
            mask = np.array(Image.open(os.path.join(masks_path,folder_id[idx].zfill(4),str(img_id)+extension)))

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
            break
        break
    return dataset_dicts


def getMask(mask_path,img_path):
    mask = cv2.imread(os.path.join(mask_path,"0000/000000.png"))
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    img = cv2.imread(os.path.join(img_path,"0000/000000.png"))
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(mask2, contours, -1, (255, 255, 255), 1)
    #Obteneidos los contornos los dibujamos en una imagen
    plt.imshow(mask2)
    plt.show()
