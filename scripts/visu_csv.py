import pandas as pd
import numpy as np
import os


FACE_DIR = os.listdir(r"../CK+/")
FACE_FILE = lambda face : f"../CK+/{face}/omlands.csv"

for dir in FACE_DIR:
    if os.path.isdir("../CK+/" + dir):
        contenu = pd.read_csv(FACE_FILE(dir), delimiter=";")

        index = contenu.iloc[:, 0]

        visages = contenu.iloc[: , 1:-1].to_numpy()
        visages = visages[:, np.newaxis]
        visages = visages.reshape((-1, 68, 2))

print(visages.shape, index.shape)
