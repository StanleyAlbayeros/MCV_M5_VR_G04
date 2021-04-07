import os
import colorama

def init_workspace(local_run, v, python_filename):
    colorama.init(autoreset=False)
    global db_path_kitti_mots
    global masks_path_kitti_mots
    global imgs_path_kitti_mots
    global db_path_mots_challenge
    global masks_path_mots_challenge
    global imgs_path_mots_challenge
    global output_path
    global pkl_path
    global train_pkl_kitti_mots
    global val_pkl_kitti_mots
    global train_pkl_mots_challenge
    global val_pkl_mots_challenge
    global thing_classes
    global mask_rcnn_models


    output_path = f"outputs/{python_filename}"
    pkl_path = "../W4/datasetpkl"
    train_pkl_kitti_mots = pkl_path + "/train_kitti_mots.pkl"
    val_pkl_kitti_mots = pkl_path + "/val_kitti_mots.pkl"
    val_pkl_mots_challenge = pkl_path + "/val_mots_challenge.pkl"
    train_pkl_mots_challenge = pkl_path + "/train_mots_challenge.pkl"
    thing_classes = ["Car", "Pedestrian", "Other"]

    if local_run:
        db_path_kitti_mots = "../resources/KITTI-MOTS"
        masks_path_kitti_mots = "../resources/KITTI-MOTS/instances"
        imgs_path_kitti_mots = "../resources/KITTI-MOTS/training/image_02"
        db_path_mots_challenge = "../resources/MOTSChallenge"
        masks_path_mots_challenge = "../resources/MOTSChallenge/instances"
        imgs_path_mots_challenge = "../resources/MOTSChallenge/train/images"
    else:
        db_path_kitti_mots = "../datasets/KITTI-MOTS"
        masks_path_kitti_mots = "../datasets/KITTI-MOTS/instances"
        imgs_path_kitti_mots = "../datasets/KITTI-MOTS/training/image_02"
        db_path_mots_challenge = "../datasets/MOTSChallenge"
        masks_path_mots_challenge = "../datasets/MOTSChallenge/instances"
        imgs_path_mots_challenge = "../datasets/MOTSChallenge/train/images"


    if not os.path.exists(output_path):
        if v:
            print(colorama.Fore.MAGENTA + "Creating output dir")
        os.makedirs(output_path)
    if not os.path.exists(pkl_path):
        if v:
            print(colorama.Fore.MAGENTA + "Creating pkl dir")
        os.makedirs(pkl_path)
    
    mask_rcnn_models ={
        "R50-C4_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
        "R50-DC5_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
        "R50-FPN_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "R50-C4_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        "R50-DC5_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        "R50-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "R101-C4_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        "R101-DC5_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
        "R101-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "X101-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml" 
    }

