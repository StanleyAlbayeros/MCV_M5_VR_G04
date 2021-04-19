import imageio
import os
import glob
from pygifsicle import optimize
from tqdm import tqdm
from pathlib import Path

class didict(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

path = f"outputs/task_a/imgs/coco_mod/mix"

images_in_dir = sorted(os.listdir(path))
img_name_list = didict()

for filename in images_in_dir:
    print(filename)
    no_ext_fname = Path(f"{filename}").stem
    no_ext_fname = no_ext_fname.split(sep="_")
    if f"{no_ext_fname[0]}" in img_name_list:
        img_name_list[no_ext_fname[0]] += 1
    else:
        img_name_list[no_ext_fname[0]] = 1

# for k, v in img_name_list.items():
#     print(f"{k} - {v}")

print(img_name_list)

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