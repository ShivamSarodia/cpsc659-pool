import os
import sys
import math
import cv2
import numpy as np

class TableDetector:
    def __init__(self):
        self.img = None
        self.imagePath = None
        self.tableEdges = None
        self.pockets = None
        self.tableColorRange = (np.array([235, 155, 0], dtype="uint8"), np.array([255, 210, 0], dtype="uint8"))

    def __log_error(self, error_str):
        raise Exception("TableDetector: " + error_str)

    def __display_image_internal(self, image_to_disp, title="image"):
        cv2.imshow(title, image_to_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image(self, image_path):
        if not os.path.isfile(image_path):
            __log_error("Image file not found")
            return
        self.image_path = image_path
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    def detect_table_edges(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return
        mask = cv2.inRange(self.img, self.tableColorRange[0], self.tableColorRange[1])
        output = cv2.bitwise_and(self.img, self.img, mask = mask)
        edges = cv2.Canny(output, 50, 150, apertureSize = 3)

        image_copy = np.copy(self.img)
        lines = cv2.HoughLines(edges,1,np.pi/180,100)
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image_copy,(x1,y1),(x2,y2),(0,0,255),2)

        self.__display_image_internal(image_copy, title="Table edges")

    def display_image(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return
        cv2.imshow('PoolTable', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    td = TableDetector()
    td.load_image("sample_table.png")
    td.detect_table_edges()()

if __name__ == '__main__':
    main()
