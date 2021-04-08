import random
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2
import os
import colorama


def generate_sample_imgs(
    target_metadata,
    validation_dataset,
    output_path,
    predictor,
    scale,
    num_imgs,
    model_name,
):
    # for d in random.sample(validation_dataset, num_imgs):
    for d in validation_dataset:
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

        tmpname = d["file_name"]
        tmpname_list = tmpname.split(os.sep)
        filepath = f"{output_path}/{model_name}/inf_images/{tmpname_list[-2]}"
        

        if not os.path.exists(filepath):
            if v:
                print(colorama.Fore.MAGENTA + f"\t\tCreating img {filepath} dir")
            os.makedirs(filepath, exist_ok=True)

        filename = f"{filepath}/{imgid}.png"
        if v:
            print(f"\t\t\tSaving image to : {filepath}/{imgid}.png")
        cv2.imwrite(filename, img)

        window_name = "image"

        # Using cv2.imshow() method
        # Displaying the image
        cv2.imshow(window_name, img)

        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(1)

        # closing all open windows
    cv2.destroyAllWindows()
