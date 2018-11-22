import os
import sys
import math
import cv2
import numpy as np

from load_game_window import load_game_window

img = load_game_window("sample_screenshot_3.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hues = hsv[:,:,0].copy()
circles = cv2.HoughCircles(hues, cv2.HOUGH_GRADIENT, 1,
                           minDist=17,
                           param1=15,
                           param2=10,
                           minRadius=15,
                           maxRadius=23)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2)
    cv2.circle(img, (i[0],i[1]), 2, (0,0,255), 3)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()