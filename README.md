# Facial Expression Classification Project 👨‍💻:

## Language
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## LIBS
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## OS
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)

## License
![License](https://img.shields.io/badge/License-MIT-green)


In the scope of our project, we will attempt to create a classifier capable of recognizing six facial expressions:

1. :smile: happy
2. :fearful: fear
3. :open_mouth: surprise
4. :angry: anger
5. 😵‍💫: disgust
6. :disappointed: sadness

We will use the 68 facial landmarks that delineate specific regions of the face contained in .csv files.

## Steps of Our Project :mag:

### Step 1: Characterizing Expressions :dart:

Our primary objective is to predict the `emotion` column in the `emotion.csv` file based on the facial points available in the `SXXX/omlands.csv` files.

To characterize an expression, we can employ various approaches, including:

1. Consider the coordinates of the points.
2. Consider the movement of the points between the neutral and apex emotion images.

Faces may appear at different locations in the image, and they may vary in size. It might be interesting to attempt face alignment so that faces are aligned with respect to the eyes and nose (stable points irrespective of the expression). In our experiments, we will use points as they are or place them in a common frame of reference (as illustrated below). We will try both approaches.

### Step 2: Handling Imbalance :balance_scale:

In a second phase, as the dataset is highly imbalanced, we will construct a new version of the dataset that ensures a balance between the number of instances for each expression. We will assess if the results vary significantly from the initial configuration.

### Step 3: Study of Occlusions and Noises :dark_sunglasses:

Occlusion results in the unavailability of a certain subset of points. For example, an occlusion of the left eye results in the absence of points (37 to 42), and an occlusion of the mouth results in the absence of points (49-68). Noise can also be artificially added to certain points by altering their values.

In a third phase, we will examine performance in the presence of facial occlusions and noises. The main question is to what extent the learning techniques proposed in the previous questions are robust to these alterations and noises in terms of extent and intensity.

#### Step 3.1: Creating Occlusions and Noises :see_no_evil:

We will code different occlusions and different noises, starting from small regions around significant elements such as the eyes, eyebrows, and mouth, and progressing to larger occlusions that hide half of the face.

#### Step 3.2: Evaluating Robustness :chart_with_upwards_trend:

We will answer the main question above by relying on quantification, i.e., through appropriate measurements.

---

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Developed by Pierre LAGUE and François MULLER (@franzele21) at the University of Lille, France. 🚀📊

