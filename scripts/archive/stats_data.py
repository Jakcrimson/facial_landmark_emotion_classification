"""
This script is used to see the of what the data is made of.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import numpy as np
from .center_points import mean_center_face

FACES_DIR_PATH  = r"CK+_lands/CK+/"
EMOTIONS_FILE   = r"CK+_lands/CK+/emotion.csv"
#[PL] - added r-strings for windows interpretation
EMOTIONS_INDEX  = ["happy", "fear", "surprise", 
                   "anger", "disgust", "sadness"] 


emotions = pd.read_csv(EMOTIONS_FILE, delimiter=";", header=0)
faces_dir = os.listdir(FACES_DIR_PATH)

def get_difference_between_base_and_apex_from_file(face_dir, session_name, face_file):
    
    # retrieve all the visages of the file
    visage = pd.read_csv(f'{face_dir}{session_name}/{face_file}', delimiter=";")

    # separate the file and relevante data (stored in visage)
    file, visage = visage.iloc[:, 0].to_numpy(), visage.iloc[:, 1:-1].to_numpy()

    base = visage[0]
    apex = visage[-1]
    diff = apex + (base-apex)
    print(diff)
    # reshapes the data, so each row is a file, and the data is grouped by two, for x and y coordinates
    base = base.reshape((-1, 68, 2))
    apex = apex.reshape((-1, 68, 2))

    centered_base = mean_center_face(base[0], display_face=True)
    centered_apex = mean_center_face(apex[0], display_face=True)
    plt.plot(diff)
# Add labels and a legend
plt.xlabel('X Difference')
plt.ylabel('Y Difference')
plt.legend()



def get_number_of_instances():
######################################################################
# Number of instances for each emotions


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




def get_number_of_lines_per_emotions():
######################################################################
# Number of total lines per emotions
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




def get_percentage_of_labelled_data():
######################################################################
# Percentage of labelled data

    labelled_faces = set(emotions.iloc[:, 0])
    labelled_percent = len(labelled_faces) * 100 / (len(labelled_faces) + len(faces_dir) - 1)
    print(f"Percentage of labelled data: {labelled_percent}")


