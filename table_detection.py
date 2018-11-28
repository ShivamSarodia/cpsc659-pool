import os
import sys
import math
import random

import cv2
import numpy as np
from ball_inference import BallClassifier

class TableDetector:
    def __init__(self):
        # Path of the image loaded into the table detector.
        self.imagePath = None

        # OpenCV image loaded into the table detector.
        self.img = None

        # OpenCV image of the game window
        self.gameWindow = None

        # The top-left coordinate of the game window relative to the entire original image
        self.gameWindowTopLeft = None, None

        # The corners of the table as a dictionary of the form {"tl": (x, y), "br": (x, y), ...}.
        # Coordinates are relative to game window.
        self.tableCorners = None

        # The original image cropped to include only the table surface.
        self.tableCrop = None

        # The top-left coordinate of table crop relative to the entire original image.
        self.tableCropTopLeft = None, None

        # Size of tableCrop in pixels as (width, height)
        self.tableSize = None, None

        # The location of the table pockets. Coordinates are relative to cropped table.
        self.pockets = None

        # List of (x, y) coordinate pairs for detected balls. Coordinates are relative to cropped table.
        self.balls = {"white": (None, None), "black": (None, None), "stripes": [], "solids": []}

        # The radius of the balls, in pixels.
        self.ball_radius = None

        # Color ranges used for detection.
        self.tableSurfaceColorRange = (np.array([150, 110, 0], dtype="uint8"), np.array([205, 205, 90], dtype="uint8"))
        self.tableWoodColorRange = (np.array([45, 49, 150], dtype="uint8"), np.array([66, 73, 175], dtype="uint8"))
        self.tableColorRange = (np.array([195, 150, 65], dtype="uint8"), np.array([255, 210, 75], dtype="uint8"))
        self.pocketColorRange = (np.array([0, 0, 0], dtype="uint8"), np.array([0, 0, 40], dtype="uint8"))
        self.windowColorRange = (np.array([37, 25, 18], dtype="uint8"), np.array([58, 38, 29], dtype="uint8"))

        # Misc parameters
        self.nms_rho_tol = 50
        self.nms_theta_tol = np.pi/180.0 * 30.0
        self.eps = 0.1
        # self.bc = BallClassifier('ball_classification_norm_params.joblib', 'ball_gbm.joblib')

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
        self.img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)

    def detect_game_window(self):
        mask = cv2.inRange(self.img, self.windowColorRange[0], self.windowColorRange[1])
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        self.gameWindow = self.img[y:y+h, x:x+w]
        self.gameWindowTopLeft = x, y

    def __line_non_max_suppression(self, lines, max_size):
        '''
        Returns non-max-suppressed list of lines from the given set of lines,
        which is sorted in order of confidence. The first line is always
        included, and upto (max_size-1) more lines may be added
        '''
        edge_lines = [lines[0][0]]
        for l in lines:
            rho, theta = l[0]

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

        mask = cv2.inRange(self.gameWindow, self.pocketColorRange[0], self.pocketColorRange[1])
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.maxArea = 10000

        params.filterByCircularity = True
        params.minCircularity = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
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

        self.pockets = {}

        # Four pockets are at the corners
        for lab in self.tableCorners:
            self.pockets[lab] = (self.tableCorners[lab][0] - self.tableCorners["tl"][0],
                                 self.tableCorners[lab][1] - self.tableCorners["tl"][1])

        # Middle pockets are projections onto the corresponding table edge.
        self.pockets['ml'] = (0, filtered_keypoints[0][1] - self.tableCorners["tl"][1])
        self.pockets['mr'] = (self.pockets["tr"][0], filtered_keypoints[1][1] - self.tableCorners["tl"][1])

    def _detect_corners(self, img, color_range):
        mask = cv2.inRange(img, color_range[0], color_range[1])
        output = cv2.bitwise_and(img, img, mask=mask)
        edges = cv2.Canny(output, 50, 150, apertureSize = 3)

        image_copy = np.copy(img)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        # non-max suppression, since lines are in order of confidence
        edge_lines = self.__line_non_max_suppression(lines, 4)
        table_corners = []
        for i in range(0, len(edge_lines)):
            for j in range(i+1, len(edge_lines)):
                table_corners.extend(self.__intersection(edge_lines[i], edge_lines[j]))
        return self.__sort_corners(table_corners)

    def detect_table_edges(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return

        wood_corners = self._detect_corners(self.gameWindow, self.tableWoodColorRange)

        y1 = wood_corners["tl"][0] + 30
        x1 = wood_corners["tl"][1] + 30
        y2 = wood_corners["br"][0] - 30
        x2 = wood_corners["br"][1] - 30

        wood_img = self.gameWindow[x1:x2,y1:y2,...]
        temp_corners = self._detect_corners(wood_img, self.tableColorRange)

        self.tableCorners = {}
        for corner_name in temp_corners:
            self.tableCorners[corner_name] = (temp_corners[corner_name][0] + y1,
                                              temp_corners[corner_name][1] + x1)

        self.tableCrop = self.gameWindow[
            self.tableCorners['tl'][1]:self.tableCorners['bl'][1],
            self.tableCorners['tl'][0]:self.tableCorners['tr'][0]].copy()
        self.tableCropTopLeft = (self.tableCorners['tl'][0] + self.gameWindowTopLeft[0],
                                 self.tableCorners['tl'][1] + self.gameWindowTopLeft[1])
        self.tableSize = self.tableCrop.shape[1], self.tableCrop.shape[0]

    def detect_balls(self):
        hsv = cv2.cvtColor(self.tableCrop, cv2.COLOR_BGR2HSV)
        hues = hsv[:,:,0].copy()

        circles = cv2.HoughCircles(hues, cv2.HOUGH_GRADIENT, 1,
                                minDist=17,
                                param1=15,
                                param2=10,

                                # 15 is less accurate ball positioning, but less likely to miss a ball
                                # 16 is more accurate ball position, but as a higher chance of missing a ball
                                minRadius=15,
                                maxRadius=19)

        full_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        sats = full_hsv[:,:,1].copy()
        
        radii = []
        for circle in circles[0]:
            is_ball = self._classify_ball(*circle, sats)
            if is_ball:
                radii.append(circle[2])

        self.ball_radius = np.max(radii)

    def _classify_ball(self, x, y, r, sats):
        """Given the coordinates of a ball, add it to the self.balls dictionary."""

        mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        circle_x = int(x + self.tableCropTopLeft[0])
        circle_y = int(y + self.tableCropTopLeft[1])
        circle_r = int(r)
        cv2.circle(mask, (circle_x, circle_y), circle_r, 1, thickness=-1)

        masked_sats = cv2.bitwise_and(sats, sats, mask=mask)
        mean_sat = np.sum(masked_sats) / np.sum(mask > 0)

        if mean_sat < 5:
            self.balls["black"] = (x, y)
            return True
        elif mean_sat < 35:
            self.balls["white"] = (x, y)
            return True

        # TODO: classify between solids, stripes, and non-balls
        is_solid = True
        is_stripe = False

        if is_solid:
            self.balls["solids"].append((x, y))
            return True
        elif is_stripe:
            self.balls["stripes"].append((x, y))
            return True
        else:
            return False

    def display_table_detections(self):
        image_copy = np.copy(self.tableCrop)

        if self.pockets:
            for pocket in self.pockets:
                cv2.circle(image_copy, self.pockets[pocket], 10, (0, 255, 0), thickness=-1)

        if self.balls["white"]:
            x, y = self.balls["white"]
            cv2.circle(image_copy, (int(x), int(y)), int(self.ball_radius), (0, 255, 0))

        if self.balls["black"]:
            x, y = self.balls["black"]
            cv2.circle(image_copy, (int(x), int(y)), int(self.ball_radius), (0, 0, 255))

        for x, y in self.balls["stripes"]:
            cv2.circle(image_copy, (int(x), int(y)), int(self.ball_radius), (255, 0, 0))

        for x, y in self.balls["solids"]:
            cv2.circle(image_copy, (int(x), int(y)), int(self.ball_radius), (255, 0, 255))

        self.__display_image_internal(image_copy, title="Table detections")

    def display_image(self):
        if self.image_path == None:
            __log_error("No image loaded")
            return
        cv2.imshow('PoolTable', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_all(self):
        self.detect_game_window()
        self.detect_table_edges()
        self.detect_pockets()
        self.detect_balls()

    def produce_classification_data(self):
        image_copy = np.copy(self.img)
        for x, y, r in self.tentative_balls:
            mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)

            circle_x = int(x + self.tableCropTopLeft[0])
            circle_y = int(y + self.tableCropTopLeft[1])
            circle_r = int(r)
            cv2.circle(mask, (circle_x, circle_y), circle_r, (255, 255, 255), thickness=-1)

            masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)
            ball_img = masked_img[circle_y - circle_r:circle_y + circle_r, circle_x - circle_r:circle_x + circle_r]
            cv2.imwrite("ball_imgs/" + str(random.randint(0, 1e10)) + ".png", ball_img)

def main():
    td = TableDetector()
    td.load_image("screenshots/screen_1022506300.png")
    td.detect_all()

    td.display_table_detections()


if __name__ == '__main__':
    main()
