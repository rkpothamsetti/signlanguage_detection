import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)

offset=30
imgSize=300

folder="datastorage/O"
counter=0

while True:
    success, img =cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand=hands[0]
        X,Y,w,h = hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[Y-offset:Y+h+offset, X-offset:X+w+offset]
        
        if imgCrop.size != 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h/w
        
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
        
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
            
            
            
    cv2.imshow("Image", img)
    key= cv2.waitKey(1)
    if key==ord("s"):
         counter += 1
         cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
         print(counter)
   