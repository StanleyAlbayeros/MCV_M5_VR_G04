import getDicts
import argparse
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
import cv2
import os

def init_path_vars(local_run: bool = False):
    global db_path
    global masks_path
    global imgs_path


    if local_run:
        db_path = "../resources/KITTI-MOTS/testing/image_02"
        masks_path = "../resources/KITTI-MOTS/instances"
        imgs_path = "../resources/KITTI-MOTS/training/image_02"
    else:
        db_path = "../datasets/KITTI-MOTS"
        masks_path = "../datasets/KITTI-MOTS/instances"
        imgs_path = "../datasets/KITTI-MOTS/training/image_02"


def parse_args():
    parser = argparse.ArgumentParser(description='Task b')
    parser.add_argument('--local', dest='local', action='store_true')
    parser.set_defaults(local=False)
    return parser.parse_args()

if __name__ == "__main__":
    parser = parse_args()
    local = parser.local
    init_path_vars(local)

    #data = getDicts.getMask(masks_path,imgs_path)
    """data = getDicts.get_dicts(db_path,imgs_path,masks_path)
    img = cv2.imread(os.path.join(masks_path,"0000/000000.png"))
    for d in data:
        for s in d["annotations"]:
            for c in s["segmentation"]:
                cv2.drawContours(img, c, -1, (255, 0, 0), 3)"""
    train,val = getDicts.split_data_kitti_motts(db_path)
    data_train = getDicts.get_dicts(train,imgs_path,masks_path)
    print(data_train)
