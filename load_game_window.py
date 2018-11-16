import os
import sys
import math
import cv2
import numpy as np

WINDOW_COLOR_MIN = np.array([37, 25, 18], dtype="uint8")
WINDOW_COLOR_MAX = np.array([58, 38, 29], dtype="uint8")

def loadGameWindow(screenshot_path):
    """From the given screenshot, produce OpenCV image containing the game window."""
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    mask = cv2.inRange(img, WINDOW_COLOR_MIN, WINDOW_COLOR_MAX)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    return img[y:y+h, x:x+w]

if __name__ == '__main__':
    window_img = loadGameWindow("sample_screenshot.png")
    cv2.namedWindow('PoolTable', cv2.WINDOW_NORMAL)
    cv2.imshow('PoolTable', window_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()