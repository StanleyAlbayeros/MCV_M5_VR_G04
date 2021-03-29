import getDicts


LOCAL_RUN = False
DB_PATH = "../datasets/KITTI-MOTS"
MASKS_PATH = "../datasets/KITTI-MOTS/instances"
IMGS_PATH = "../datasets/KITTI-MOTS/training/image_02"

if LOCAL_RUN:
    DB_PATH = "../resources/KITTI-MOTS/testing/image_02"
    MASKS_PATH = "../resources/KITTI-MOTS/instances"
    IMGS_PATH = "../resources/KITTI-MOTS/training/image_02"

data = getDicts.getMask(MASKS_PATH,IMGS_PATH)
#data = getDicts.get_dicts(DB_PATH,IMGS_PATH,MASKS_PATH)
