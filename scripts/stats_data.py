"""
This script is used to see the of what the data is made of.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm

FACES_DIR_PATH  = r"CK+_lands/CK+/"
EMOTIONS_FILE   = r"CK+_lands/CK+/emotion.csv"
#[PL] - added r-strings for windows interpretation
EMOTIONS_INDEX  = ["happy", "fear", "surprise", 
                   "anger", "disgust", "sadness"]


######################################################################
# Number of instances for each emotions

emotions = pd.read_csv(EMOTIONS_FILE, delimiter=";", header=0)

nb_ex_emotions = {}
for line in range(len(emotions)):
    tmp_emotion = EMOTIONS_INDEX[emotions.iloc[line, :]["emotion"]-1]

    if tmp_emotion not in nb_ex_emotions:
        nb_ex_emotions[tmp_emotion] = 0

    nb_ex_emotions[tmp_emotion] += 1

pprint(nb_ex_emotions)

plt.pie([nb_ex_emotions[x] for x in EMOTIONS_INDEX], 
        labels=EMOTIONS_INDEX, autopct='%1.0f%%')
plt.title("Number of instances of each class in emotion.csv")
plt.show()


######################################################################
# Number of total lines per emotions

faces_dir = os.listdir(FACES_DIR_PATH)
# pprint(faces_files)

nb_line_emotions = {}
for face in tqdm(faces_dir):
    face_path = os.path.join(FACES_DIR_PATH, face)

    if os.path.isdir(face_path):
        face_file = os.path.join(face_path, "omlands.csv")

        face_content = pd.read_csv(face_file, delimiter=";")
        nb_record = {}
        for line in range(len(face_content)):
            record_nb = face_content.iloc[line, 0]
            
            if record_nb not in nb_record:
                nb_record[record_nb] = 0

            nb_record[record_nb] += 1
        
        for line in range(len(emotions)):
            file = emotions.iloc[line, 1]
            emotion_face = emotions.iloc[line, 0]

            if emotion_face == face:
                tmp_emotion = EMOTIONS_INDEX[emotions.iloc[line, :]["emotion"]-1]

                if tmp_emotion not in nb_line_emotions:
                    nb_line_emotions[tmp_emotion] = 0

                nb_line_emotions[tmp_emotion] += nb_record[file]

pprint(nb_line_emotions)

plt.pie([nb_line_emotions[x] for x in EMOTIONS_INDEX], 
        labels=EMOTIONS_INDEX, autopct='%1.0f%%')
plt.title("Number of instances of each class in all the different session files")
plt.show()


######################################################################
# Percentage of labelled data

labelled_faces = set(emotions.iloc[:, 0])
labelled_percent = len(labelled_faces) * 100 / (len(labelled_faces) + len(faces_dir) - 1)
print(f"Percentage of labelled data: {labelled_percent}")
