import getDicts
import numpy as np
import os
import io_tools


db_path = "../datasets/MOTSChallenge"
img_path = "../datasets/MOTSChallenge/train/images"

db_path2 = "../datasets/KITTI-MOTS"
masks_path2 = "../datasets/KITTI-MOTS/instances"
img_path2 = "../datasets/KITTI-MOTS/training/image_02"


pkl_path = "../W4/datasetpkl"
train_pkl = pkl_path + "/train.pkl"
val_pkl = pkl_path + "/val.pkl"

def split_data(base_path):
    training = ["2", "6", "7", "8", "10", "13", "14", "16", "18"]
    training = np.char.zfill(training, 4)

    train_dataset = {}
    val_dataset = {}

    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        if file[0:-4] in training:
            train_dataset[f"{file[0:-4]}"] = annotations
        else:
            val_dataset[f"{file[0:-4]}"] = annotations
    return train_dataset,val_dataset

#mots-challenge
"""
train = getDicts.split_data_mots_challenge(db_path,img_path,train_pkl,val_pkl)

print(train)"""

##kitti-mots
train,val = split_data(db_path2)
train_data = getDicts.get_dicts(train,img_path2,".png")
print(train_data)


"""data_train = getDicts.get_dicts(train,img_path)
print(data_train)"""