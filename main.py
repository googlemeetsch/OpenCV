import cv2
import numpy as np
import face_recognition
import os

path = 'face_images'
images = []
knownNames = []
knownFace = []
dir = os.listdir(path)
lock = {}

for cl in dir:
    if not cl.startswith('.'):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        knownNames.append(os.path.splitext(cl)[0])
        knownFace.append(curImg)

for name in knownNames:
    lock[name] = False
    print(name, lock[name])


def findEncodings(images):
    encodeList = []
    for i in images:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodedFaces = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    curFaceFrame = face_recognition.face_locations(imgSmall)
    curEncoded = face_recognition.face_encodings(imgSmall, curFaceFrame)

    for encodeFace, facePos in zip(curEncoded, curFaceFrame):
        matches = face_recognition.compare_faces(encodedFaces, encodeFace)
        faceDis = face_recognition.face_distance(encodedFaces, encodeFace)

        minIdx = np.argmin(faceDis)

        if matches[minIdx]:
            name = knownNames[minIdx]
            if lock[name] is False:
                lock[name] = True
                print(name, lock[name])
            y1, x2, y2, x1 = facePos
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{name}, {lock[name]}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)