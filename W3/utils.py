import cv2
import os
import io_tools
import pycocotools
from tqdm import tqdm
import cv2
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt
import numpy as np

def print_percent_done(index, total, bar_len=50, title='Please wait'):
    '''
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

    if round(percent_done) == 100:
        print('\t✅')
"""
Obtención de las boxes de cada imagen (KITTI-MOTS)
"""
def get_dicts(base_path,images_path,extension=".png"):
    raw_dicts = []
    dataset_dicts = []
    Pedestrians = []
    Cars =[]
    folder_id = []
    for file in sorted(os.listdir(base_path + "/instances_txt")):
        annotations = io_tools.load_txt(base_path + "/instances_txt/" + file)
        raw_dicts.append(annotations)
        folder_id.append(file[:-4])
    for idx, dicts in tqdm(enumerate(raw_dicts)):
        for key,value in dicts.items():
            record = {}
            img_id = str(key).zfill(6)
            img_path = os.path.join(images_path,folder_id[idx].zfill(4),str(img_id)+extension)
            # print(img_path)
            img = cv2.imread(img_path)
            height,width,channels = img.shape

            record["file_name"] = img_path
            record["image_id"] = img_id
            record["height"] = height
            record["width"] = width
            objs = []
            for instance in value:
                category_id = instance.class_id
                if category_id == 1 or category_id == 2:
                    bbox = pycocotools.mask.toBbox(instance.mask)
                    obj = {
                        "bbox": [float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": category_id -1,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def split_data(dicts):
    tr_s = int(np.floor(len(dicts) * 0.6))
    val_s = int(np.floor(len(dicts) * 0.8))
    train_ds = dicts[:tr_s]
    val_ds = dicts[tr_s + 1: val_s]
    test_ds = dicts[val_s + 1:]

    return train_ds, val_ds, test_ds
    
def no_split_data(dicts):
    tr_s = int(np.floor(len(dicts) * 0.6))
    val_s = int(np.floor(len(dicts) * 0.8))
    train_ds = dicts[:tr_s]
    val_ds = dicts[tr_s + 1: val_s]
    test_ds = dicts[val_s + 1:]

    return train_ds, val_ds, test_ds


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    overlay = img.copy()
    output = img.copy()
    alpha = 0.9
    cv2.rectangle(overlay, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(overlay, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness, )
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def write_text_two(img1, img2, str1, str2):
    txt_font = cv2.FONT_HERSHEY_PLAIN
    fontsize = 5
    fontcolor = (255,0,255)
    bg_color = (200,200,200)
    fontthickness = 3
    txtSize1 = cv2.getTextSize(str1, txt_font, fontsize, fontthickness)[0]
    txtSize2 = cv2.getTextSize(str2, txt_font, fontsize, fontthickness)[0]
    margin = 10
    textX1 = img2.shape[1] - (txtSize1[0])- margin
    textY1 = margin
    textX2 = img2.shape[1] - (txtSize2[0])- margin
    textY2 = margin
    coords1 = (textX1,textY1)
    coords2 = (textX2,textY2)
    # print(coords)
    

    img1 = draw_text(img1, str1, txt_font, coords1, fontsize, fontthickness, fontcolor, bg_color)
    img2 = draw_text(img2, str2, txt_font, coords2, fontsize, fontthickness, fontcolor, bg_color)
    return img1, img2

