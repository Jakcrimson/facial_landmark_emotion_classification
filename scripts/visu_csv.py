import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

FACE_DIR = os.listdir(r"CK+_lands/CK+/")
FACE_FILE = lambda face : f"CK+_lands/CK+/{face}/omlands.csv"
EMOTIONS_FILE   = r"CK+_lands/CK+/emotion.csv"
emotions = pd.read_csv(EMOTIONS_FILE, delimiter=";", header=0)
EMOTIONS_INDEX  = ["happy", "fear", "surprise", 
                   "anger", "disgust", "sadness"]



def plot_facial_landmarks(session_data, expression_name):
    """Facial landmark display for a given session and a given expression

    Args:
        session_data : the session
        expression_name : the corresponding emotion
    """
    x_coords = []
    y_coords = []

    for i in range(0, 136, 2):
        x_col_name = f'Coord_{i}'
        y_col_name = f'Coord_{i + 1}'
        print(session_data[x_col_name])
        x_coords.append(session_data[x_col_name])
        y_coords.append(session_data[y_col_name])

    for i in range(len(x_coords)):
        plt.figure()
        plt.scatter(x_coords[:-1], y_coords[:-1])
        plt.title(f"Expression: {expression_name}, Image {i + 1}")
        plt.gca().invert_yaxis()
        plt.show()



"""Go through the emotion.csv file and extract the session name
    build the file path from it to get the session and the coordinates.
"""
#for index, row in emotions.iterrows():
#    subject = row['subject']
#    session = row['file']
#    expression = row['emotion']
#
#    omlands_filename =  f"CK+_lands/CK+/{subject}/omlands.csv"
#
#    session_data = pd.read_csv(omlands_filename, delimiter=';', header=None, names=['file'] + [f'x{i}' for i in range(1, 69)])
#
#    session_data = session_data[session_data['file'] == session]
#
#    #plot_facial_landmarks(session_data, f"Expression {expression}")




for dir in FACE_DIR:
    if os.path.isdir("CK+_lands/CK+/" + dir):
        contenu = pd.read_csv(FACE_FILE(dir), delimiter=";")

        index = contenu.iloc[:, 0]
        visages = contenu.iloc[: , 1:-1].to_numpy()
        visages = visages[:, np.newaxis]
        visages = visages.reshape((-1, 68, 2))

print(visages.shape, index.shape)



#### TESTING, ATTENTION PLEASE ####
omlands_filename =  r"CK+_lands/CK+/S010/omlands.csv"
session_data = pd.read_csv(omlands_filename, delimiter=';', names=['file'] + [f'Coord_{i}' for i in range(0, 136)])
plot_facial_landmarks(session_data.tail(1), EMOTIONS_INDEX[5])