import cmake as cmake
import cv2
import numpy as np
import face_recognition
import os

path = 'face_images'
images = []
KnownNames = []
Knownface = []
dir = os.listdir(path)
print(dir)

for cl in dir:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    KnownNames.append(os.path.splitext(cl)[0])
    Knownface.append(curImg)
