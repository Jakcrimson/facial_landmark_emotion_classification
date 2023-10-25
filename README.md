# Projet de Classification des Expressions Faciales :woman_technologist:

## Langage
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## LIBS
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## OS
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)

## Licence
![License](https://img.shields.io/badge/License-MIT-green)

Dans le cadre de ce projet, vous allez essayer de réaliser un classifieur capable de reconnaître six expressions faciales : 

1. :smile: happy
2. :fearful: fear
3. :open_mouth: surprise
4. :angry: anger
5. :face_vomiting: disgust
6. :disappointed: sadness

Vous utiliserez les 68 points faciaux qui délimitent des regions spécifiques du visage.

Les données sont issues du dataset `CK+` contenant des personnes mimant chacune une série des expressions. Vous trouverez dans le fichier `ck+_lands.zip` l'ensemble des fichiers `csv` décrivant le dataset. 

## Premières approches :mag:

L’objectif premier est d’essayer de prédire la colonne `emotion` du fichier `emotion.csv` en s'appuyant sur les points faciaux disponibles dans les fichiers `SXXX/omlands.csv`.

Pour caractériser une expression vous pouvez utilisez différentes approches parmis lesquelles

1. :dart: considérer les coordonnées des points
2. :arrow_double_up: considérer le déplacement des points entre l'image neutre et l'image apex

Les visages peuvent se trouver à différents endroits dans l'image. Ils peuvent également être de tailles différentes. Il peut être intéressant d'essayer d'aligner les visages de telle sorte que les visages soient alignés par rapport aux yeux et au nez (points stables indépendamment de l'expression). Ainsi, dans vos expérimentations, vous pourrez utiliser les points tels quels sont ou bien en les remettant dans un repère commun (comme illustré ci-dessous). Essayez les deux approches.

## Deuxième approche: gestion du déséquilibre :balance_scale:

Dans un deuxième temps, comme le dataset est très déséquilibré, vous construirez une nouvelle version du dataset qui garantit l'équilibre entre le nombre d'instances de chaque expression. Est-ce que les résultats varient beaucoup par rapport à la configuration initiale ? Commentez.

## Troisième approche : les occultations :dark_sunglasses:

Une occultation résulte dans l'indisponibilité d'un certain sous-ensemble des points. Par exemple, une occultation de l'oeil gauche se traduit par l'absence des points (37 à 42) et une occultation de la bouche se traduit par l'absence de points (49-68). Du bruit peut être également ajouté artificiellement à certains points en modifiant leur valeur.

Dans un troisième temps, vous étudierez les performances en présence d'occultations et bruitages faciaux. La question principale est de savoir jusqu'à quel point les techniques d'apprentissage proposées dans les questions précédentes sont robustes à ces altérations et bruitages en termes d'étendue et d'intensité. 

1. :see_no_evil: Codez différentes occultations et différents bruitages en partant des petites regions autour des éléments significatifs tels que les yeux, les sourcils, la bouche et en allant jusqu'à des larges occultations cachant la moitié du visage.
2. :chart_with_upwards_trend: Répondez à la question principale ci-dessus en vous appuyant sur une quantification, c'est-à-dire par des mesures adéquates.
