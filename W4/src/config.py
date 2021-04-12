import os
import colorama

def init_workspace(v=False, _python_filename=""):
    colorama.init(autoreset=False)
    global db_path_kitti_mots
    global masks_path_kitti_mots
    global imgs_path_kitti_mots
    global db_path_mots_challenge
    global masks_path_mots_challenge
    global imgs_path_mots_challenge
    global output_path
    global txt_results_path
    global gen_img_path
    global pkl_path
    global pkl_train_path
    global pkl_val_path
    global training_pkl
    global validation_pkl
    global train_pkl_kitti_mots
    global val_pkl_kitti_mots
    global train_pkl_mots_challenge
    global val_pkl_mots_challenge
    global train_combo
    global val_combo
    global thing_classes
    global mask_rcnn_models
    global mask_rcnn_results
    global cityscapes_models
    global cityscapes_results
    global python_filename
    global verbose
    global thing_colors



    python_filename = _python_filename
    verbose = v
    pkl_path = "datasetpkl"
    pkl_train_path = f"{pkl_path}/train"
    pkl_val_path = f"{pkl_path}/val"

    train_pkl_kitti_mots = f"{pkl_train_path}/train_kitti_mots.pkl"
    train_pkl_mots_challenge = f"{pkl_train_path}/train_mots_challenge.pkl"
    training_pkl = f"{pkl_train_path}/training.pkl"

    val_pkl_kitti_mots = f"{pkl_val_path}/val_kitti_mots.pkl"
    val_pkl_mots_challenge = f"{pkl_val_path}/val_mots_challenge.pkl"
    validation_pkl = f"{pkl_val_path}/validation.pkl"

    output_path = f"outputs/{python_filename}"
    txt_results_path = f"{output_path}/txt_results" 
    gen_img_path = f"{output_path}"    

    if not os.path.exists(output_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {output_path}")
        os.makedirs(output_path)

    if not os.path.exists(txt_results_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {txt_results_path}")
        os.makedirs(txt_results_path)

    if not os.path.exists(gen_img_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {gen_img_path}")
        os.makedirs(gen_img_path)

    if not os.path.exists(pkl_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {pkl_path}")
        os.makedirs(pkl_path)

    if not os.path.exists(pkl_train_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {pkl_train_path}")
        os.makedirs(pkl_train_path)

    if not os.path.exists(pkl_val_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {pkl_val_path}")
        os.makedirs(pkl_val_path)



    thing_classes = ["Person", "Other", "Car"]
    thing_colors = [(50,255,50), (102,255,255), (255,50,255)]

    base_dir = "../resources"
    db_path_kitti_mots = f"{base_dir}/KITTI-MOTS"
    db_path_mots_challenge = f"{base_dir}/MOTSChallenge"
    masks_path_kitti_mots = f"{db_path_kitti_mots}/instances"
    imgs_path_kitti_mots = f"{db_path_kitti_mots}/training/image_02"
    masks_path_mots_challenge = f"{db_path_mots_challenge}/instances"
    imgs_path_mots_challenge = f"{db_path_mots_challenge}/train/images"

    mask_rcnn_models = {
        # "R50-FPN_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
        "R50-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        # "R101-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        # "X101-FPN_x3" : "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        # "R50-DC5_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
        "R50-DC5_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
        # "R101-DC5_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
        # "R50-C4_x1" : "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
        # "R50-C4_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
        # "R101-C4_x3" : "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
        "City-R50-FPN" : "Cityscapes/mask_rcnn_R_50_FPN.yaml",
        
    }
    mask_rcnn_results = {
        # "R50-FPN_x1" : "",
        "R50-FPN_x3" : "", #######
        # "R101-FPN_x3" : "",
        # "X101-FPN_x3" : "",
        # "R50-DC5_x1" : "",
        "R50-DC5_x3" : "",
        # "R101-DC5_x3" : "",
        # "R50-C4_x1" : "",
        # "R50-C4_x3" : "",
        # "R101-C4_x3" : "",
        "City-R50-FPN" : "",
    }

    cityscapes_models = {
        "R50-FPN" : "Cityscapes/mask_rcnn_R_50_FPN.yaml"
    }
    cityscapes_results = {
        "R50-FPN" : ""
    }

def create_txt_results_path(target_path):
    if not os.path.exists(target_path):
        if verbose:
            print(colorama.Fore.MAGENTA + f"Creating {target_path}")
        os.makedirs(target_path)
