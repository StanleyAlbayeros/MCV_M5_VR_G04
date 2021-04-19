import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import colorama

from src import config


def build_cooc_matrix(pred_dict):

    df = pd.DataFrame(pred_dict).set_index("Class_names")
    df = df.T

    save_dataframe(df, "_predict_results")

    cooc_mat = df_to_cooc(df)

    return cooc_mat


def df_to_cooc(df):

    df = df.loc[:, (df != 0).any(axis=0)]

    low_thrsh = 100
    high_thrsh = 3

    # ###### "High" co-occurrence ####
    # for col in df.columns:
    #     if len(df[col].unique()) <= high_thrsh:
    #         df.drop(col,inplace=True,axis=1)

    # ##### "Low" co-occurrence ####
    # for col in df.columns:
    #     if len(df[col].unique()) >= low_thrsh:
    #         df.drop(col, inplace=False, axis=1)

    ###### "middle" co-occurrence ####
    # for col in df.columns:
    #     if len(df[col].unique()) <= low_thrsh:
    #         if len(df[col].unique()) >= high_thrsh:
    #             df.drop(col,inplace=True,axis=1)

    # print(df)

    df_asint = df.astype(int)
    cooc_mat = df_asint.T.dot(df_asint)
    save_dataframe(cooc_mat, "_cooc_mat")
    return cooc_mat


def save_dataframe(dataframe, add_str=""):
    dataframe.to_csv(config.csv_save_path(add_str), encoding="utf-8", index=True)


def load_cooc_from_csv():
    cooc_csv = config.csv_save_path("_cooc_mat")
    if config.verbose:
        print(colorama.Fore.MAGENTA + f"Reading cooc matrix from {cooc_csv}")
    return pd.read_csv(cooc_csv, index_col=0)


def save_cooc_plot(cooc_mat):
    my_dpi = 200
    plt.figure(figsize=(2000 / my_dpi, 1800 / my_dpi), dpi=my_dpi)

    sns.set_theme()
    sns.set(font_scale=0.8)
    sns.heatmap(cooc_mat, annot=True, fmt="d", linewidths=.2,cbar=False)
    # plt.show()

    plt.tight_layout()
    plt.autoscale()
    if config.verbose:
        print(colorama.Fore.CYAN + f"Saving plot to {config.plt_save_path()}")
    plt.savefig(config.plt_save_path())


def cooc_plot_selective(cooc_mat):

    sns.set_theme(style="whitegrid")
    my_dpi = 200
    plt.figure(figsize=(2000 / my_dpi, 1800 / my_dpi), dpi=my_dpi)

    sns.heatmap(cooc_mat, annot=True, fmt="d",  )
    
    plt.savefig(config.plt_save_path())


