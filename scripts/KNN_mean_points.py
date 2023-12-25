"""
KNN classifier, using the CK+_mean dataset
The CK+_mean dataset is one file per emotions, and in it there is the 
mean of all points of all faces for this emotion.
"""

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

DATASET_PATH = "CK+_lands/CK+_mean"

# load the dataset
df = pd.DataFrame(columns=["data", "target"])
for file in os.listdir(DATASET_PATH):
    file_content = pd.read_csv(os.path.join(DATASET_PATH, file), delimiter=";", header=None)
    print(file_content)

    file = int(file[:-4])
    df.loc[len(df.index)] = [file_content, file]


knn = KNeighborsClassifier(n_neighbors=1)