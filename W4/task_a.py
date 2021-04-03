import getDicts
import argparse
import os


def init_path_vars(local_run: bool = False):

    global db_path
    global masks_path
    global imgs_path
    global output_path

    output_path = "outputs/task_a"

    if local_run:
        db_path = "../resources/KITTI-MOTS/testing/image_02"
        masks_path = "../resources/KITTI-MOTS/instances"
        imgs_path = "../resources/KITTI-MOTS/training/image_02"
    else:
        db_path = "../datasets/KITTI-MOTS"
        masks_path = "../datasets/KITTI-MOTS/instances"
        imgs_path = "../datasets/KITTI-MOTS/training/image_02"

    if not os.path.exists(output_path):
        print("Creating output dir")
        os.makedirs(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task a: Apply pre-trained Mask-RCNN models to KITTI-MOTS validation set"
    )
    parser.add_argument("--local", dest="local", action="store_true")
    parser.set_defaults(local=False)
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    local = parser.local
    init_path_vars(local)

    data = getDicts.getMask(masks_path, imgs_path, output_path)
    print(data)
    # data = getDicts.get_dicts(db_path,imgs_path,masks_path)
