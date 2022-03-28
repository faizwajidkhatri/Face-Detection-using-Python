import cv2
from config import  *
from  uitils import *
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path= FACE_PATH


onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i] # face/user1.jpg
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)


# Linear Binary Phase Histogram Classifier
model = cv2.face.LBPHFaceRecognizer_create()
""" this line will generate error run the following command 

python -m pip install --user opencv-contrib-python

"""

model.train(np.asarray(Training_Data), np.asarray(Labels))
print('Model Training Complete !!!')





cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + '% Confidence it is USER'

        cv2.putText(image, display_string, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 55, 255), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()