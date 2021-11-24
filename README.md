#code of students attendance face recognition using ml python and Ai
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
from numpy.lib.function_base import _digitize_dispatcher
import pandas as pd 

# from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(roll_no,name):
    with open('Attendance.csv', 'a') as f:
        myDataList = pd.read_csv('Attendance.csv')
        csvwriter = csv.writer(f)       
        if roll_no not in myDataList.Roll_no.values:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S') 
            csvwriter.writerow([roll_no,name,dtString])


encodeListKnown = findEncodings(images)
print('Encoding Complete')
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            img_name = classNames[matchIndex].upper()
            img_name_list = img_name.split('_')
            name = img_name_list[1]+' '+img_name_list[2]
            roll_no = int(img_name_list[0])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(roll_no,name)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
