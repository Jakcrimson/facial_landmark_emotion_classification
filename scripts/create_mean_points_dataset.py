"""
Create the dataset of the mean points for each emotions 
Will be used for KNN

Just run the script to create the new dataset with the mean face points per emotions
Be sure to be in the right directory!
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

# dictionnary that will contain all faces of one emotions
all_emotions = {int(emotion): [] for emotion in emotion_index["emotion"].unique()}
for emotion in tqdm(emotion_index["emotion"].unique()):         # looping through all emotions

    # fetch the rows that has the same emotions as the one we are looping through
    emotion_df = emotion_index.loc[:, ["subject", "file"]].where(emotion_index["emotion"] == emotion).dropna()

    for line in emotion_df.iterrows():                          # looping through all rows that have the same emotions
        subject, file = line[1]["subject"], int(line[1]["file"])

        # fetch the face points from the dataset
        subject_face = pd.read_csv(os.path.join(INPUT_FACE_DIR, subject, "omlands.csv"), header=None, delimiter=";")

        # fetch only the points from the good file (as in the variable of line 28)
        # fetch the last one, because it is the one with the right emotion
        subject_face = subject_face.where(subject_face.iloc[:, 0] == file).dropna().iloc[-1, 1:]

        if len(subject_face) > 0:
            subject_face = subject_face.to_numpy().reshape((68, 2))
            all_emotions[emotion].append(subject_face)              # append the points to the dictionnary

# calculate the mean of the points for each emotions
for emotion in all_emotions.keys():
    all_emotions[emotion] = np.sum(all_emotions[emotion], axis=0) / len(all_emotions[emotion])

pprint(all_emotions)

# # OPTIONNAL : diplay the mean faces per emotion
# for emotion in all_emotions.keys():
#     plt.scatter(all_emotions[emotion][:, 0], -all_emotions[emotion][:, 1])
#     plt.show()


# save the mean faces per emotion, with first column being the x, the second y (so one line = one point)
if not os.path.exists(OUTPUT_FACE_DIR):
    os.mkdir(OUTPUT_FACE_DIR)
for emotion in all_emotions.keys():
    np.savetxt(os.path.join(OUTPUT_FACE_DIR, f"{emotion}.csv"), all_emotions[emotion].reshape(68, 2), delimiter=";")

