import getDicts
import numpy as np
import os
import io_tools
from src import config
import getDicts


pkl_path = "../W4/datasetpkl"
train_pkl_kitti_mots = f"{pkl_path}/train/train_kitti_mots.pkl"
val_pkl_kitti_mots = f"{pkl_path}/val/val_kitti_mots.pkl"
val_pkl_mots_challenge = f"{pkl_path}/val/val_mots_challenge.pkl"
train_combo = f"{pkl_path}/train/train_combo.pkl"
val_combo = f"{pkl_path}/val/val_combo.pkl"
train_pkl_mots_challenge = f"{pkl_path}/train/train_mots_challenge.pkl"
thing_classes = ["Person", "Other", "Car"]

# txt_results_path = f"{output_path}/txt_results"
base_dir = "../resources"
db_path_kitti_mots = f"{base_dir}/KITTI-MOTS"
db_path_mots_challenge = f"{base_dir}/MOTSChallenge"
masks_path_kitti_mots = f"{db_path_kitti_mots}/instances"
imgs_path_kitti_mots = f"{db_path_kitti_mots}/training/image_02"
masks_path_mots_challenge = f"{db_path_mots_challenge}/instances"
imgs_path_mots_challenge = f"{db_path_mots_challenge}/train/images"

# getDicts.generate_motschallenge_pkls(
#     db_path_mots_challenge,
#     imgs_path_mots_challenge,
#     train_pkl_mots_challenge,
#     val_pkl_mots_challenge,
#     True,
# )
config.init_workspace(True, "lul")

getDicts.generate_combined_pkl(pkl_path)

