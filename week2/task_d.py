"""

Tutorial Train!

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


IMG_PATH = "/home/mcv/datasets/KITTI/data_object_image_2/"
DATABASE_PATH = "/home/mcv/datasets/KITTI/"
categories = ["Car","Van","Truck","Pedestrian","Person_sitting","Cyclist","Tram","Misc","DontCare"]


def get_kitti_dicts(img_dir,filenames_dir,mode):
    with open(filenames_dir+mode+"_kitti.txt") as filename:
        filenames = filename.readlines()

    dataset_dicts = []
    for idx,filename in enumerate(filenames):
        record = {}
        id_image = filename[:-5]
        if mode == "train":
            folder = "training"
        else:
            folder = "training"
        filename = os.path.join(img_dir+folder+"/image_2/",id_image + ".png")

        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = id_image
        record["height"] = height
        record["width"] = width

        with open(os.path.join(filenames_dir + folder + "/label_2/"+id_image+".txt")) as annotations:
            annos = annotations.readlines()


        objs = []
        for det in annos:
            to_list = det.split(" ")
            obj = {
                "bbox" : [to_list[4],to_list[5],to_list[6],to_list[7]],
                "bbox_mode" : BoxMode.XYXY_ABS,
                "category_id" : to_list[0]
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("kitti_" + d, lambda d=d: get_kitti_dicts(IMG_PATH,DATABASE_PATH,"train"))
    MetadataCatalog.get("kitti_" + d).set(thing_classes=categories)
kitti_metadata = MetadataCatalog.get("kitti_train")
print("Training")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kitti_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1200    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print("End of training")

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset_dicts = get_kitti_dicts(IMG_PATH,DATABASE_PATH,"train")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=kitti_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
