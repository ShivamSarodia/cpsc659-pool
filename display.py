import cv2
import numpy as np

class Display():
    def __init__(self, img):
        self.img = np.copy(img)

    def show_pockets(self, pockets):
        for pocket in pockets:
            cv2.circle(self.img, pockets[pocket], 10, (0, 255, 0), thickness=-1)

    def show_tent_balls(self, tentative_balls):
        for x, y, r in tentative_balls:
            cv2.circle(self.img, (int(x), int(y)), int(r), (0, 255, 0))

    def add_circle(self, coords, radius=10, color=(0, 255, 0), thickness=1):
        cv2.circle(self.img, coords, radius, color, thickness=thickness)

    def show(self):
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()