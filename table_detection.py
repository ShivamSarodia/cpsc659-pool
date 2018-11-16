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
        self.tableCorners = None
        self.pockets = None
        self.tableColorRange = (np.array([235, 155, 0], dtype="uint8"), np.array([255, 210, 0], dtype="uint8"))
        self.nms_rho_tol = 50
        self.nms_theta_tol = np.pi/180.0 * 30.0
        self.eps = 0.1

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

    def __line_non_max_suppression(self, lines, max_size):
        '''
        Returns non-max-suppressed list of lines from the given set of lines,
        which is sorted in order of confidence. The first line is always
        included, and upto (max_size-1) more lines may be added
        '''
        edge_lines = [lines[0][0]]
        for rho, theta in lines[0]:
            is_suppressed = False
            for rho_p, theta_p in edge_lines:
                if math.fabs(rho - rho_p) < self.nms_rho_tol and math.fabs(theta - theta_p) < self.nms_theta_tol:
                    is_suppressed = True
                    break
            if not is_suppressed:
                edge_lines.append((rho, theta))
            if len(edge_lines) == max_size:
                break
        return edge_lines

    def __intersection(self, line1, line2):
        '''Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        '''
        rho1, theta1 = line1
        rho2, theta2 = line2
        if math.fabs(theta1 - theta2) <= self.eps:
            return []
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [(x0, y0)]

    def detect_table_edges(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return
        mask = cv2.inRange(self.img, self.tableColorRange[0], self.tableColorRange[1])
        output = cv2.bitwise_and(self.img, self.img, mask = mask)
        edges = cv2.Canny(output, 50, 150, apertureSize = 3)

        image_copy = np.copy(self.img)
        lines = cv2.HoughLines(edges,1,np.pi/180,100)

        # non-max suppression, since lines are in order of confidence
        edge_lines = self.__line_non_max_suppression(lines, 4)
        self.tableCorners = []
        for edge1 in edge_lines:
            for edge2 in edge_lines:
                self.tableCorners.extend(self.__intersection(edge1, edge2))
        for corner in self.tableCorners:
            cv2.circle(image_copy, corner, 10, (0, 255, 0))

        for rho,theta in edge_lines:
            # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image_copy,(x1,y1),(x2,y2),(0,0,255),2)
        
        self.tableEdges = edge_lines
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
