import os
import re
from ast import literal_eval
from collections import OrderedDict

import numpy as np
from numpy import nan
import pandas as pd
import csv
import json

# This string is slightly different from your sample which had an extra bracket
<<<<<<< HEAD
RESULTS_PATH = "../task_a_txt/"


def read_file(file_path):
    f = open(file_path), "r")
    line = f.read()
    match = re.search(r"^OrderedDict\((.+)\)\s*$", line)
=======
RESULTS_PATH="../task_a_txt/"
def read_file(file):
    f = open(os.path.join(RESULTS_PATH, file), "r")
    line = f.readline()
    match = re.search(r'^OrderedDict\((.+)\)\s*$', line)
>>>>>>> main
    data = match.group(1)
    # This allows safe evaluation: data can only be a basic data structure
    return OrderedDict(eval(data))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert txts to tables")
    parser.add_argument('-path', type=dir_path)


onlyfiles = [
    f for f in os.listdir(RESULTS_PATH) if os.path.isfile(os.path.join(RESULTS_PATH, f))
]
col = [
    "Method",
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AP-Person",
    "AP-Other",
    "AP-Car",
]

<<<<<<< HEAD
fp = open("bbox.csv", "w")
writer = csv.writer(fp, delimiter="\t")
writer.writerow(col)

fp1 = open("segm.csv", "w")
writer1 = csv.writer(fp1, delimiter="\t")
=======
fp=open(os.path.join(RESULTS_PATH, "bbox.csv"), "w")
writer = csv.writer(fp, delimiter='\t')
writer.writerow(col)

fp1=open(os.path.join(RESULTS_PATH, "segm.csv"), "w")
writer1 = csv.writer(fp1, delimiter='\t')
>>>>>>> main
writer1.writerow(col)

for file in onlyfiles:
    content = read_file(file)
<<<<<<< HEAD
    writer.writerow([file.replace(".txt", "")] + list(content["bbox"].values()))
    writer1.writerow([file.replace(".txt", "")] + list(content["segm"].values()))
=======
    auxb=[]
    auxs=[]
    for i in list(content["bbox"].values()):
        auxb.append(str(i).replace(".",","))
    for i in list(content["segm"].values()):
        auxs.append(str(i).replace(".",","))

    writer.writerow([file.replace(".txt", "")]+list(auxb))
    writer1.writerow([file.replace(".txt", "")]+list(auxs))
>>>>>>> main
