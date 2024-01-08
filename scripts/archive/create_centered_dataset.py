"""
This script is used to create the centered dataset, using the function of center_points.py
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from center_points import mean_center_face

FACE_DIR_ORI    = r"CK+_lands/CK+/"
FACE_DIR_NEW    = r"CK+_lands/CK+_centered/"
FACE_FILE_NAME  = "omlands.csv"

if os.path.exists(FACE_DIR_NEW):
    raise Exception("The new directory already exists")


os.mkdir(FACE_DIR_NEW)

for face in tqdm(os.listdir(FACE_DIR_ORI)):
    face_dir = os.path.join(FACE_DIR_ORI, face)
    if os.path.isdir(face_dir):
        new_face_dir = os.path.join(FACE_DIR_NEW, face)
        os.mkdir(new_face_dir)

        face_file = os.path.join(face_dir, FACE_FILE_NAME)
        content = pd.read_csv(face_file, delimiter=";", header=None)

        # placeholder line, to have the right dimension
        new_content = np.empty((1, 137), dtype=np.float64)
        for i, line in enumerate(content.to_numpy()):
            data = line[1: -1].reshape((68, 2))
            data = mean_center_face(data).flatten()

            new_line = np.hstack((line[0], data))
            new_content = np.vstack([new_content, new_line])
        
        # we ignore the placeholder line
        new_content = pd.DataFrame(new_content[1:])

        new_content.to_csv(os.path.join(new_face_dir, FACE_FILE_NAME), sep=";", header=None, index=None)
