from tqdm import tqdm
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ORIGIN_DATASET  = "CK+_lands/CK+_centered"
EMOTIONS_INDEX  = "CK+_lands/CK+/emotion.csv"
IMAGES_DIR      = "CK+_lands/images"
EMOTIONS        = {
    1: "happy",
    2: "fear",
    3: "surprise",
    4: "anger",
    5: "disgust",
    6: "sadness"
}


# delete all old files and directories
for root, dirs, files in os.walk(IMAGES_DIR, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
if os.path.exists(IMAGES_DIR):
    os.rmdir(IMAGES_DIR)
# make the new directories
os.mkdir(IMAGES_DIR)
for split_name in ["train", "val", "test"]:
    os.mkdir(os.path.join(IMAGES_DIR, split_name))
    for emotion_name in list(EMOTIONS.values()):
        os.mkdir(os.path.join(IMAGES_DIR, split_name, emotion_name))


# retrieve the emotions dataset
emotions = pd.read_csv(EMOTIONS_INDEX, header=0, delimiter=";")

# creation of the dataset split (here 70-20-10)
dataset_split = np.arange(len(emotions))
X_train, X_test = train_test_split(dataset_split, test_size=10)
X_train, X_val  = train_test_split(X_train, test_size=20)

# create all the faces 
for emo_id in tqdm(emotions.index[4:]):
    tmp_subject, tmp_file, tmp_emotion = emotions.loc[emo_id, :]
    tmp_points = pd.read_csv(os.path.join(ORIGIN_DATASET, tmp_subject, "omlands.csv"), header=None, delimiter=";")
    tmp_points = tmp_points.where(tmp_points.iloc[:, 0] == tmp_file).dropna()
    tmp_points = tmp_points.iloc[-1, 1:]

    direction = "train" if emo_id in X_train else "val" if emo_id in X_val else "test"

    # saving the faces
    plt.scatter(tmp_points[::2], -tmp_points[1::2], c="black")
    plt.axis("off")
    plt.savefig(os.path.join(IMAGES_DIR, direction, EMOTIONS[tmp_emotion], f"{tmp_subject}.png"), bbox_inches='tight')
    plt.clf()