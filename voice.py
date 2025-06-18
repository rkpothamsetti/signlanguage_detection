import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3  # Voice synthesis
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

offset = 30
imgSize = 300
previous_label = ""
last_spoken_time = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        X, Y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[Y-offset:Y+h+offset, X-offset:X+w+offset]

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

            # Speak only if label changes or after a delay
            if label != previous_label or (time.time() - last_spoken_time > 2):
                print("Speaking:", label)
                engine.say(label)
                engine.runAndWait()
                previous_label = label
                last_spoken_time = time.time()

            # Draw annotations
            cv2.putText(imgOutput, label, (X, Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (X - offset, Y - offset), (X + w + offset, Y + h + offset), (0, 255, 0), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
