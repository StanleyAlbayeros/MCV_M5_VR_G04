import imageio
import os
import glob
from pygifsicle import optimize
from tqdm import tqdm

base_path_list = [
    "outputs/generate_vid_from_model/COCO_KITTI/R50-FPN_x3/PNG/0007",
    "outputs/generate_vid_from_model/COCO_KITTI/City-R50-FPN/PNG/0007",
    "outputs/generate_vid_from_model/COCO_KITTI/R50-DC5_x3/PNG/0007",
    "outputs/generate_vid_from_model/COCO_KITTI_MOTSC/R50-FPN_x3/PNG/0007",
    "outputs/generate_vid_from_model/COCO_KITTI_MOTSC/City-R50-FPN/PNG/0007",
    "outputs/generate_vid_from_model/COCO_KITTI_MOTSC/R50-DC5_x3/PNG/0007",
]

for path in tqdm(base_path_list, desc=f"Creating gifs", colour="Cyan"):

    path_parts = path.split(os.sep)

    gif_path = f"outputs/GIF/{path_parts[2]}/{path_parts[3]}/GIF/{path_parts[5]}.gif"
    optimized_gif_path = (
        f"outputs/GIF/{path_parts[2]}/{path_parts[3]}/GIF/{path_parts[5]}_optimized.gif"
    )
    gif_path_dir = f"outputs/GIF/{path_parts[2]}/{path_parts[3]}/GIF"
    if not os.path.exists(gif_path_dir):
        os.makedirs(gif_path_dir)

    print(f"saving gif to {gif_path}")
    with imageio.get_writer(gif_path, mode="I", fps=30, subrectangles=True) as writer:
        for subdir, _, files in os.walk(path):
            for file in tqdm(files, desc=f"Creating {path_parts[5]}.gif", colour="Magenta"):
                writer.append_data(imageio.imread(os.path.join(subdir, file)))

    optimize(gif_path, optimized_gif_path)  # For creating a new one
