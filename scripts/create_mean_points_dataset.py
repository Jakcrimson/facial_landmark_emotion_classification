"""
Create the dataset of the mean points for each emotions 
Will be used for KNN
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm

INPUT_FACE_DIR      = "CK+_lands/CK+_centered"
OUTPUT_FACE_DIR     = "CK+_lands/CK+_mean"
EMOTIONS_INDEX_FILE = "CK+_lands/CK+/emotion.csv"

# fetch the labels
emotion_index = pd.read_csv(EMOTIONS_INDEX_FILE, header=0, delimiter=";")

all_emotions = {int(emotion): [] for emotion in emotion_index["emotion"].unique()}
for emotion in tqdm(emotion_index["emotion"].unique()):
    emotion_df = emotion_index.loc[:, ["subject", "file"]].where(emotion_index["emotion"] == emotion).dropna()
    for line in emotion_df.iterrows():
        subject, file = line[1]["subject"], int(line[1]["file"])
        subject_face = pd.read_csv(os.path.join(INPUT_FACE_DIR, subject, "omlands.csv"), header=None, delimiter=";")
        subject_face = subject_face.where(subject_face.iloc[:, 0] == file).dropna().iloc[-1, 1:]

        if len(subject_face) > 0:
            subject_face = subject_face.to_numpy().reshape((68, 2))
            all_emotions[emotion].append(subject_face)

for emotion in all_emotions.keys():
    all_emotions[emotion] = np.sum(all_emotions[emotion], axis=0) / len(all_emotions[emotion])

pprint(all_emotions)

# for emotion in all_emotions.keys():
#     plt.scatter(all_emotions[emotion][:, 0], -all_emotions[emotion][:, 1])
#     plt.show()
if not os.path.exists(OUTPUT_FACE_DIR):
    os.mkdir(OUTPUT_FACE_DIR)
for emotion in all_emotions.keys():
    np.savetxt(os.path.join(OUTPUT_FACE_DIR, f"{emotion}.csv"), all_emotions[emotion].reshape(1, -1), delimiter=";")

