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

def findEncodings(images):
    encodeList= []
    for i in images:
        img = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodedFaces = findEncodings(images)
print('Encoding Complete')
