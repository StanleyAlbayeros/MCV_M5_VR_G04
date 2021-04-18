import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config


def build_cooc_matrix(pred_dict):

    df = pd.DataFrame(pred_dict).set_index("Class_names")
    df = df.T

    save_dataframe(df, "raw")
    
    cooc_mat = df_to_cooc(df)

    return cooc_mat

def df_to_cooc(df):
    
    df = df.loc[:, (df != 0).any(axis=0)]

    ###### "High" co-occurrence ####

    # for col in df.columns:
    #     if len(df[col].unique()) <= 3:
    #         df.drop(col,inplace=True,axis=1)

    ###### "Low" co-occurrence ####
    for col in df.columns:
        if len(df[col].unique()) >= 100:
            df.drop(col, inplace=True, axis=1)

    # ###### "middle" co-occurrence ####
    # for col in df.columns:
    #     if len(df[col].unique()) >= 5:
    #         df.drop(col,inplace=True,axis=1)

    # print(df)
    df_asint = df.astype(int)
    cooc_mat = df_asint.T.dot(df_asint)
    return cooc_mat


def save_dataframe(dataframe, add_str=""):
    dataframe.to_csv(config.csv_save_path(add_str), encoding="utf-8", index=True)


def save_cooc_plot(cooc_mat):
    my_dpi = 200
    plt.figure(figsize=(2000 / my_dpi, 1800 / my_dpi), dpi=my_dpi)

    sns.set_theme()
    sns.heatmap(cooc_mat, annot=True, fmt="d")
    # plt.show()

    plt.tight_layout()
    plt.autoscale()
    plt.savefig(config.plt_filename)
