import os
import sys
import math
import cv2
import cv2.cv as cv
import numpy as np
from ball_inference import BallClassifier

from load_game_window import load_game_window

class TableDetector:
    def __init__(self):
        self.img = None
        self.imagePath = None
        self.tableEdges = None
        self.tableCorners = None
        self.pockets = None
        self.balls = None
        self.ballCropRadius = 9
        self.tableSurfaceColorRange = (np.array([150, 110, 0], dtype="uint8"), np.array([205, 205, 90], dtype="uint8"))
        self.tableColorRange = (np.array([195, 150, 0], dtype="uint8"), np.array([255, 210, 25], dtype="uint8"))
        self.pocketColorRange = (np.array([0, 0, 0], dtype="uint8"), np.array([0, 0, 40], dtype="uint8"))
        self.nms_rho_tol = 50
        self.nms_theta_tol = np.pi/180.0 * 30.0
        self.eps = 0.1
        self.bc = BallClassifier('ball_classification_norm_params.joblib', 'ball_gbm.joblib')

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
        # self.img = load_game_window(self.image_path)
        self.img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

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

    def __sort_corners(self, corner_list):
        print(corner_list)
        assert len(corner_list) == 4

        sorted_corners = {}
        corner_list.sort()
        sorted_corners['tl'] = corner_list[0]
        sorted_corners['tr'] = corner_list[2]
        sorted_corners['bl'] = corner_list[1]
        sorted_corners['br'] = corner_list[-1]

        return sorted_corners


    def detect_pockets(self):
        '''
        Detect the table pockets. Four of the positions are already known,
        so only need to detect the 2 middle pockets
        '''

        mask = cv2.inRange(self.img, self.pocketColorRange[0], self.pocketColorRange[1])
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.maxArea = 10000

        params.filterByCircularity = True
        params.minCircularity = 0.5

        detector = cv2.SimpleBlobDetector(params)
        keypoints = detector.detect(255 - mask)
        filtered_keypoints = []
        for keypoint in keypoints:
            if keypoint.pt[0] >= self.tableCorners['tl'][0]-20 and \
                keypoint.pt[0] < self.tableCorners['br'][0]+20 and \
                keypoint.pt[1] >= self.tableCorners['tl'][1] and \
                keypoint.pt[1] < self.tableCorners['br'][1]:

                # could modify this to add projections of points onto edges
                # instead of the raw points themselves
                filtered_keypoints.append((int(keypoint.pt[0]), int(keypoint.pt[1])))

        filtered_keypoints.sort()
        self.pockets = dict(self.tableCorners)
        self.pockets['ml'] = filtered_keypoints[0]
        self.pockets['mr'] = filtered_keypoints[1]

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
        table_corners = []
        for i in range(0, len(edge_lines)):
            for j in range(i+1, len(edge_lines)):
                table_corners.extend(self.__intersection(edge_lines[i], edge_lines[j]))
        self.tableCorners = self.__sort_corners(table_corners)
        self.tableEdges = edge_lines

    def detect_balls(self):
        table_crop = self.img[
            self.tableCorners['tl'][1]:self.tableCorners['bl'][1],
            self.tableCorners['tl'][0]:self.tableCorners['tr'][0]].copy()
        mask_lower = cv2.bitwise_and(table_crop, table_crop, mask=255 - cv2.inRange(table_crop, self.tableSurfaceColorRange[0], self.tableSurfaceColorRange[1]))
        # self.__display_image_internal(mask_lower)
        cimg = cv2.cvtColor(mask_lower, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(cimg, cv.CV_HOUGH_GRADIENT, 1.0, minDist=6, param1=25, param2=10, minRadius=6, maxRadius=10)
        image_copy = np.copy(table_crop)
        circles = np.uint16(np.around(circles))

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in circles[0,:]:
            # draw the outer circle
            # print(i[1]-self.ballCropRadius, i[1]+1+self.ballCropRadius, i[0]-self.ballCropRadius, i[0]+1+self.ballCropRadius)
            # print(image_copy.shape)
            ball_crop = image_copy[i[1]-self.ballCropRadius:i[1]+1+self.ballCropRadius, i[0]-self.ballCropRadius:i[0]+1+self.ballCropRadius, ...]
            pred = self.bc.classify_ball(ball_crop)
            if not pred == 2:
                cv2.circle(image_copy,(i[0],i[1]),i[2],colors[pred],2)
                # draw the center of the circle
                cv2.circle(image_copy,(i[0],i[1]),2,(0,0,255),3)
        self.__display_image_internal(image_copy)


    def display_table_detections(self):
        image_copy = np.copy(self.img)
        if self.tableCorners:
            tl = self.tableCorners['tl']
            bl = self.tableCorners['bl']
            tr = self.tableCorners['tr']
            br = self.tableCorners['br']

            cv2.line(image_copy, tl, tr, (0, 0, 255), 2)
            cv2.line(image_copy, tl, bl, (0, 0, 255), 2)
            cv2.line(image_copy, tr, br, (0, 0, 255), 2)
            cv2.line(image_copy, bl, br, (0, 0, 255), 2)

        # if self.tableEdges:
        #     for rho,theta in self.tableEdges:
        #         # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a*rho
        #         y0 = b*rho
        #         x1 = int(x0 + 1000*(-b))
        #         y1 = int(y0 + 1000*(a))
        #         x2 = int(x0 - 1000*(-b))
        #         y2 = int(y0 - 1000*(a))
        #
        #         cv2.line(image_copy,(x1,y1),(x2,y2),(0,0,255),2)

        if self.pockets:
            for pocket in self.pockets:
                cv2.circle(image_copy, self.pockets[pocket], 10, (0, 255, 0))

        self.__display_image_internal(image_copy, title="Table detections")

    def display_image(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return
        cv2.imshow('PoolTable', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    td = TableDetector()
    td.load_image("sample_table_2.png")
    td.detect_table_edges()
    td.detect_pockets()
    td.detect_balls()

if __name__ == '__main__':
    main()
