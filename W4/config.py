import os
import colorama

def init_workspace(local_run, v, python_filename):
    colorama.init(autoreset=False)
    global db_path
    global masks_path
    global imgs_path
    global output_path
    global pkl_path
    global train_pkl
    global val_pkl
    global thing_classes
    global mask_rcnn_models

    output_path = f"outputs/{python_filename}"
    pkl_path = "datasetpkl"
    train_pkl = pkl_path + "/train.pkl"
    val_pkl = pkl_path + "/val.pkl"
    thing_classes = ["Car", "Pedestrian", "Other"]

    if local_run:
        db_path = "../resources/KITTI-MOTS"
        masks_path = "../resources/KITTI-MOTS/instances"
        imgs_path = "../resources/KITTI-MOTS/training/image_02"
    else:
        db_path = "../datasets/KITTI-MOTS"
        masks_path = "../datasets/KITTI-MOTS/instances"
        imgs_path = "../datasets/KITTI-MOTS/training/image_02"

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