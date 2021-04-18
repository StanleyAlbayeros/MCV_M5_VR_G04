import os
import re
from collections import OrderedDict

import numpy as np
from numpy import nan
import csv

# This string is slightly different from your sample which had an extra bracket
RESULTS_PATH="../outputs/task_b/txt_results/COCO_KITTI_MOTSC/"
def read_file(file):
    f = open(os.path.join(RESULTS_PATH, file), "r")
    line = f.readline()
    line2 = f.readline()

    match = re.search(r'^OrderedDict\((.+)\)\s*$', line)
    data = match.group(1)

    #match = re.search(r'Time elapsed: (.+)\s*$', line2)
    #et = match.group(1)
    # This allows safe evaluation: data can only be a basic data structure
    return OrderedDict(eval(data)), line2


onlyfiles = [f for f in os.listdir(RESULTS_PATH) if os.path.isfile(os.path.join(RESULTS_PATH, f))]
col = ["", "AP", "AP50", "AP75", "APs", "APm", "APl", "AP-Person", "AP-Other", "AP-Car", "Elapsed Time", "ratio AP:Time", "ratio AP50:Time","ratio Car:Time","ratio Person:Time"]

fp = open(os.path.join(RESULTS_PATH, "task_b_results.csv"), "w")
writer = csv.writer(fp, delimiter='\t')
bboxes=[]
segments=[]
for file in onlyfiles:
    content, et = read_file(file)
    auxb=[]
    auxs=[]
    for i in list(content["bbox"].values()):
        auxb.append(str(i).replace(".",","))
    for i in list(content["segm"].values()):
        auxs.append(str(i).replace(".",","))

    times=[]
    times.append(str(et).replace(".",","))
    times.append(str(np.float_(et)/np.float_(content["bbox"]["AP"])).replace(".",","))
    times.append(str(np.float_(et)/np.float_(content["bbox"]["AP50"])).replace(".",","))
    times.append(str(np.float_(et)/np.float_(content["bbox"]["AP-Person"])).replace(".",","))
    times.append(str(np.float_(et)/np.float_(content["bbox"]["AP-Car"])).replace(".",","))

    bboxes.append([file.replace(".txt", "")]+list(auxb)+list(times))

    times = []
    times.append(str(et).replace(".", ","))
    times.append(str(np.float_(et) / np.float_(content["segm"]["AP"])).replace(".", ","))
    times.append(str(np.float_(et) / np.float_(content["segm"]["AP50"])).replace(".", ","))
    times.append(str(np.float_(et) / np.float_(content["segm"]["AP-Person"])).replace(".", ","))
    times.append(str(np.float_(et) / np.float_(content["segm"]["AP-Car"])).replace(".", ","))

    segments.append([file.replace(".txt", "")]+list(auxs)+list(times))

writer.writerow(["Bounding Boxes"])
writer.writerow(col)
for row in bboxes:
    writer.writerow(row)

writer.writerow([])

writer.writerow(["Segmentation"])
writer.writerow(col)

for row in segments:
    writer.writerow(row)
