import random
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
import os
import colorama
from src import config


def generate_gt_images(
    target_metadata,
    target_dataset,
    output_path,
    add_str,
    predictor,
    scale,
    num_imgs,
    model_name,
):
    # for d in random.sample(validation_dataset, num_imgs):
    for d in target_dataset:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=target_metadata,
            scale=scale,
            instance_mode=ColorMode.IMAGE_BW,
        )

        vis = v.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        
        imgid = d["image_id"]

        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # img = out.get_image()[:, :, ::-1]

        tmpname = d["file_name"]
        tmpname_list = tmpname.split(os.sep)
        filepath = f"{output_path}/models/{model_name}/inf_images/{tmpname_list[-2]}{add_str}"
        

        if not os.path.exists(filepath):
            if config.verbose:
                print(colorama.Fore.MAGENTA + f"\t\tCreating img {filepath} dir")
            os.makedirs(filepath, exist_ok=True)

        filename = f"{filepath}/{imgid}.png"
        if config.verbose:
            print(f"\t\t\tSaving image to : {filepath}/{imgid}.png")
        cv2.imwrite(filename, img)
""" 
        window_name = "image"

        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow(window_name, img)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(1)

        # closing all open windows
    cv2.destroyAllWindows() """



def generate_pred_images(
    target_metadata,
    target_dataset,
    output_path,
    add_str,
    predictor,
    scale,
    num_imgs,
    model_name,
):
    # for d in random.sample(validation_dataset, num_imgs):
    for d in target_dataset:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=target_metadata,
            scale=scale,
            instance_mode=ColorMode.IMAGE_BW,
        )

        # vis = v.draw_dataset_dict(d)
        # img = vis.get_image()[:, :, ::-1]
        
        imgid = d["image_id"]

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = out.get_image()[:, :, ::-1]

        tmpname = d["file_name"]
        tmpname_list = tmpname.split(os.sep)
        filepath = f"{output_path}/models/{model_name}/inf_images/{tmpname_list[-2]}{add_str}"
        

        if not os.path.exists(filepath):
            if config.verbose:
                print(colorama.Fore.MAGENTA + f"\t\tCreating img {filepath} dir")
            os.makedirs(filepath, exist_ok=True)

        filename = f"{filepath}/{imgid}.png"
        if config.verbose:
            print(f"\t\t\tSaving image to : {filepath}/{imgid}.png")
        cv2.imwrite(filename, img)
""" 
        window_name = "image"

        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow(window_name, img)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(1)

        # closing all open windows
    cv2.destroyAllWindows() """

