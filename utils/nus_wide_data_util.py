import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle

def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        # print("df shape", df.shape)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels

    # get XA, which are image low level features
    features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
    dfs = []
    if dtype=="Train":
        filenamelist= ['Train_Normalized_CM55.dat', 
                    'Train_Normalized_CH.dat',
                    'Train_Normalized_EDH.dat',
                    'Train_Normalized_CORR.dat',
                    'Train_Normalized_WT.dat' ]
    else: 
        filenamelist= ['Test_Normalized_CM55.dat', 
                    'Test_Normalized_CH.dat',
                    'Test_Normalized_EDH.dat',
                    'Test_Normalized_CORR.dat',
                    'Test_Normalized_WT.dat' ]

    for file in filenamelist:
        df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
        df.dropna(axis=1, inplace=True)
        dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]

    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], selected.values[:]
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], selected.values[:n_samples]

