import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3  # Voice synthesis
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

# Read labels
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

offset = 30
imgSize = 300
previous_label = ""
last_spoken_time = 0

current_word = ""         # Stores combined letters
last_letter_time = 0      # Time when last letter was added
letter_delay = 3          # Delay in seconds between letters

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        X, Y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[Y - offset:Y + h + offset, X - offset:X + w + offset]

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

            # Accept new letter only if it's different and enough time passed
            if label != previous_label and (time.time() - last_letter_time > letter_delay):
                current_word += label
                print("Word so far:", current_word)
                previous_label = label
                last_letter_time = time.time()

            # Draw box and label
            cv2.putText(imgOutput, label, (X, Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (X - offset, Y - offset), (X + w + offset, Y + h + offset), (0, 255, 0), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Show current word on screen
    cv2.putText(imgOutput, f"Word: {current_word}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # Press 's' to speak the word
    if key == ord('s'):
        if current_word:
            print("Speaking word:", current_word)
            engine.say(current_word)
            engine.runAndWait()
            current_word = ""  # Clear after speaking

    # Press 'b' to delete the last letter
    if key == ord('b'):
        if current_word:
            current_word = current_word[:-1]
            print("Updated word:", current_word)

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
