import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# CONSTANTES (DON'T CHANGE IT)
FACE_DIR    = r"CK+_lands/CK+/"
FACE_FILE   = "omlands.csv"

def mean_center_face(face_points: np.ndarray, center_points: np.ndarray = np.arange(27, 31), display_face: bool = False) -> np.ndarray:
    """
Used to center the points of faces according to given points with their horizontal mean

Parameters
----------
face_points: array_like
    All the points of a visage
center_points: array_like (default: np.arange(27, 31))
    Points of the visage that will be considered as the centre (usually the nose)
display_face: boolean (default: False)
    If you want to see the beautiful face, set this to True
    
Returns
-------
out: array_like
    The visage, but centered
    """
    # retrieve the coords of the center points
    center_points_coords = face_points[center_points, :]

    # is an array in the form [x, 0], where x is the mean of the abscisse value of the center points
    correction = np.array([np.mean(center_points_coords[:, 0]), 0])

    # center all the face points by sustracting it with the correction
    new_face_points = face_points - correction

    # display the visage, with the center points in red
    if display_face:
        plt.scatter(new_face_points[:, 0], -new_face_points[:, 1])
        plt.scatter(new_face_points[center_points, 0], -new_face_points[center_points, 1], c="r")
        plt.show()

    return new_face_points


# # TMP USAGE 
# # Used to display all faces, centered with the previous function
# for face_path in tqdm(os.listdir(FACE_DIR)):
#     face_path = os.path.join(FACE_DIR, face_path)

#     if os.path.isdir(face_path):
#         face = os.path.join(face_path, FACE_FILE)
#         # retrieve all the visages of the file
#         visage = pd.read_csv(face, delimiter=";")

#         # separate the file and relevante data (stored in visage)
#         file, visage = visage.iloc[:, 0].to_numpy(), visage.iloc[:, 1:-1].to_numpy()
#         # reshapes the data, so each row is a file, and the data is grouped by two, for x and y coordinates
#         visage = visage.reshape((-1, 68, 2))

#         for tmp_visage in visage:
#             mean_center_face(tmp_visage, display_face=True)

# print(visage)
# plt.scatter(visage[:, 0], -visage[:, 1])
# plt.scatter(visage[27: 31, 0], -visage[27: 31, 1], c="r")
# plt.show()