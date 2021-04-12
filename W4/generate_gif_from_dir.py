import imageio
import os
import glob
from pygifsicle import optimize
from tqdm import tqdm

image_set = "0013"
base_path_list = [
    
    # f"outputs/generate_vid_from_model/task_a/R50-FPN_x3/PNG/{image_set}",
    # f"outputs/generate_vid_from_model/COCO_KITTI/R50-FPN_x3/PNG/{image_set}",
    # f"outputs/generate_vid_from_model/COCO_KITTI/City-R50-FPN/PNG/{image_set}",
    # f"outputs/generate_vid_from_model/COCO_KITTI/R50-DC5_x3/PNG/{image_set}",
    f"outputs/generate_vid_from_model/COCO_KITTI_MOTSC/R50-FPN_x3/PNG/{image_set}",
    f"outputs/generate_vid_from_model/COCO_KITTI_MOTSC/City-R50-FPN/PNG/{image_set}",
    f"outputs/generate_vid_from_model/COCO_KITTI_MOTSC/R50-DC5_x3/PNG/{image_set}",
]

for path in tqdm(base_path_list, desc=f"Creating gifs", colour="Cyan"):

    path_parts = path.split(os.sep)
    datasets=""
    dt = path_parts[2]
    if dt == "COCO_KITTI":
        datasets = "CK"
    if dt == "COCO_KITTI_MOTSC":
        datasets = "CKM"
    


    gif_path = f"outputs/GIF/{datasets}_{path_parts[3]}_{path_parts[5]}.gif"
    optimized_gif_path = (
        f"outputs/GIF/{datasets}_{path_parts[3]}_{path_parts[5]}_optimized.gif"
    )
    gif_path_dir = f"outputs/GIF"
    if not os.path.exists(gif_path_dir):
        os.makedirs(gif_path_dir)

    print(f"saving gif to {gif_path}")
    i=0
    launchcmd = f"ffmpeg -framerate 20 -pattern_type glob -i '{path}/*.png' -r 15  -vf scale=512:-1 {gif_path}"
    os.system(launchcmd)
    """ with imageio.get_writer(gif_path, mode="I", fps=5) as writer:
        for subdir, _, files in os.walk(path):
            for file in tqdm(files, desc=f"Creating {path_parts[5]}.gif", colour="Magenta"):
                
                if i<100: writer.append_data(imageio.imread(os.path.join(subdir, file)))
                i+=1

    optimize(gif_path, optimized_gif_path)  # For creating a new one """
    


"""

ffmpeg -framerate 20 -pattern_type glob -i '*.png'  -r 15  -vf scale=512:-1  out.gif \
;




"""