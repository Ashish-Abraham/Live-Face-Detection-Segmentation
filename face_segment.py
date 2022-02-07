import cv2
import numpy as np
from masks import apply_mask,get_extended_image
from cv2 import CascadeClassifier

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
model='Models\mask_rcnn_coco.h5'

while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    for(x,y,w,h) in faces:    
        face_image = get_extended_image(img, x, y, w, h, 0.4)
        ##applying mask
        result= apply_mask(face_image,model)
        
    image = result[1]
    cv2.imshow('Segments',image)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()