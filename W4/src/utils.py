import random
from detectron2.utils.visualizer import Visualizer
import cv2


def generate_sample_imgs(
    target_metadata,
    validation_dataset,
    v,
    output_path,
    predictor,
    scale,
    num_imgs,
    model_name
):
    for d in random.sample(validation_dataset, num_imgs):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=target_metadata, scale=scale)
        vis = v.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        imgid = d["image_id"]
        filename = f"{output_path}/{model_name}/{imgid}.png"
        if v:
            print(filename)
        cv2.imwrite(filename, img)