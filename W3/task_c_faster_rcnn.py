"""

Task 2: Use object detection models in inference: Faster R-CNN

"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

from detectron2.utils.visualizer import ColorMode

from PIL import Image
import io_tools
import pycocotools
from tqdm import tqdm


"""
Calculate of IoU (move to utils file)
"""

def getIoU(predicted_bbox,gt_bbox):
    iou = ""
    return iou

"""
Obtención de las boxes de cada imagen
"""
def get_dicts(path):
    raw_dicts = []
    dataset_dicts = []
    Pedestrians = []
    Cars =[]
    for file in sorted(os.listdir(path + "/instances_txt")):
        annotations = io_tools.load_txt(path + "/instances_txt/" + file)
        raw_dicts.append(annotations)
    for idx, dicts in tqdm(enumerate(raw_dicts)):
        for key,value in dicts.items():
            record = {}
            img_id = str(key).zfill(6)
            img_path = os.path.join(IMG_PATH,str(idx).zfill(4),str(img_id)+".png")
            img = cv2.imread(img_path)
            height,width,channels = img.shape

            record["file_name"] = img_path
            record["image_id"] = img_id
            record["heigth"] = height
            record["width"] = width
            objs = []
            for instance in value:
                category_id = instance.class_id
                if category_id == 1 or category_id == 2:
                    bbox = pycocotools.mask.toBbox(instance.mask)
                    obj = {
                        "bbox": [float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": category_id,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts


LOCAL_RUN = False
IMG_PATH = "../datasets/KITTI-MOTS/training/image_02"
LABEL_PATH = "../datasets/KITTI-MOTS/instances_txt"
BASE_PATH = "../datasets/KITTI-MOTS/"

if LOCAL_RUN:
    IMG_PATH = "../resources/KITTI-MOTS/training/image_02"
    LABEL_PATH = "../resources/KITTI-MOTS/instances_txt"
    BASE_PATH = "../resources/KITTI-MOTS/"


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

for d in ["train"]:
    DatasetCatalog.register("kitti-mots_"+d, lambda d=d: get_dicts(BASE_PATH))
    MetadataCatalog.get("kitti-mots_train").set(thing_classes=["Pedestrian","Car"])
    kitti_mots_metadata = MetadataCatalog.get("kitti-mots_train")

dataset_dicts = get_dicts(BASE_PATH)

for rand in random.sample(dataset_dicts, 1):
    img = cv2.imread(rand["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kitti_mots_metadata, scale=0.5)
    print(rand)
    out = visualizer.draw_dataset_dict(rand)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
    cv2.imwrite("out_kittimotts_faster_rcnn.png", out.get_image()[:, :, ::-1])








get_dicts(BASE_PATH)
"""for folder in sorted(os.listdir(IMG_PATH)):

    for image in sorted(os.listdir(IMG_PATH + "/" + folder)):
        print("imagen "+ LABEL_PATH + "/" + folder + "/" + image+ "\n")

        label = cv2.imread(LABEL_PATH + "/" + folder + "/" + image,0)
        height, width = label.shape
        label = np.asarray(label)


        masks = list(np.unique(label))[1:-1]
        pedestrians = []
        cars = []
        print(masks)
        for mask in masks:
            # Esto obtiene una box con todas las coordenadas que forman parte de un objeto, hay que separarlas

            box = np.argwhere(label == mask)
            x1,y1 = np.min(box,axis = 0)
            x2,y2 = np.max(box,axis = 0)
            bbox = [x1,y1,x2,y2]
            if mask == 7:
                pedestrians.append(bbox)
            elif mask == 3:
                cars.append(bbox)

            # testing bbox detection
        test_image = np.zeros((height, width), dtype=np.uint8)
        for pedestrian in pedestrians:
            test_image[pedestrian[0]:pedestrian[2],pedestrian[1]:pedestrian[3]] = 122
            print(pedestrian)
            plt.imshow(test_image)
            plt.show()
            plt.imshow(label)
            plt.show()
        break"""





"""for category_folder in os.listdir(IMG_PATH):
    for filename in random.sample(os.listdir(IMG_PATH + "/" + category_folder),3):
        print("validation image: "+IMG_PATH + "/" + category_folder + "/" + filename)
        im = cv2.imread(IMG_PATH+"/"+category_folder+"/"+filename)
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()"""
