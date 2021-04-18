import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def build_cooc_matrix(pred_dict):
    df = pd.DataFrame(pred_dict).set_index("Class_names")
    df = df.T
    df_asint = df.astype(int)
    cooc_mat = df_asint.T.dot(df_asint)
    return cooc_mat