import matplotlib.pyplot as plt
import getDicts

IMG_PATH = "../datasets/KITTI-MOTS/training/image_02"
LABEL_PATH = "../datasets/KITTI-MOTS/instances_txt"
BASE_PATH = "../datasets/KITTI-MOTS/"

data = getDicts.split_data(BASE_PATH)
print(data)