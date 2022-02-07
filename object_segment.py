import cv2
import numpy as np
from masks import apply_mask

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
model='Models\mask_rcnn_coco.h5'

while True:
    ret, img = cap.read()

    ##applying mask
    result= apply_mask(img,model)
        
    image = result[1]
    cv2.imshow('Segments',image)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()