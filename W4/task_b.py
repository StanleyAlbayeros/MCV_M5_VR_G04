
# %%

import getDicts
import argparse
import matplotlib.pyplot as plt
import pycocotools.mask as rletools
import cv2
import os
# %%

def init_path_vars(local_run: bool = True):
    global db_path
    global masks_path
    global imgs_path
    global output_path

    output_path = "outputs/task_b"

    if local_run:
        db_path = "../resources/KITTI-MOTS"
        masks_path = "../resources/KITTI-MOTS/instances"
        imgs_path = "../resources/KITTI-MOTS/training/image_02"
        dataset_pkls = "datasetpkl"
    else:
        db_path = "../datasets/KITTI-MOTS"
        masks_path = "../datasets/KITTI-MOTS/instances"
        imgs_path = "../datasets/KITTI-MOTS/training/image_02"
        dataset_pkls = "datasetpkl"

    if not os.path.exists(output_path):
        print("Creating output dir")
        os.makedirs(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Task b')
    parser.add_argument('--local', dest='local', action='store_true')
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
    # %%
    train,val = getDicts.split_data_kitti_motts(db_path)
    data_train = getDicts.get_dicts(train,imgs_path,masks_path)
    data_val = getDicts.get_dicts(val,imgs_path,masks_path)
    print(len(data_train))
    print(len(data_val))

# %%
