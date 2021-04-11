import os
import re
from ast import literal_eval
from collections import OrderedDict
from numpy import nan
import pandas as pd
import csv
import json
# This string is slightly different from your sample which had an extra bracket
RESULTS_PATH="../TestResults/"
def read_file(file):
    f = open(os.path.join(RESULTS_PATH, file), "r")
    line = f.read()
    match = re.search(r'^OrderedDict\((.+)\)\s*$', line)
    data = match.group(1)
    # This allows safe evaluation: data can only be a basic data structure
    return OrderedDict(eval(data))


onlyfiles = [f for f in os.listdir(RESULTS_PATH) if os.path.isfile(os.path.join(RESULTS_PATH, f))]
col=["Method", "AP", "AP50", "AP75", "APs", "APm", "APl", "AP-Person", "AP-Other", "AP-Car"]

fp=open("bbox.csv", "w")
writer = csv.writer(fp, delimiter='\t')
writer.writerow(col)

fp1=open("segm.csv", "w")
writer1 = csv.writer(fp1, delimiter='\t')
writer1.writerow(col)

for file in onlyfiles:
    content = read_file(file)
    writer.writerow([file.replace(".txt", "")]+list(content["bbox"].values()))
    writer1.writerow([file.replace(".txt", "")]+list(content["segm"].values()))
