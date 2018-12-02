import os
import sys
import math
import argparse
import cv2
# import cv2.cv as cv
from PIL import Image
import numpy as np
import mahotas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load
# from sklearn import cross_validation, metrics

class BallClassifier:
    def __init__(self, norm_path, classifier_path):
        self.norm_path = norm_path
        self.classifier_path = classifier_path
        self.scaler = load(self.norm_path)
        self.alg = load(self.classifier_path)


    def __fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def __fd_haralick(self, image):    # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0).flatten()
        return haralick

    def __fd_histogram(self, image, mask=None):
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bins = 32
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        return hist.flatten()

    def __build_features(self, im):
        images = [im]

        # im = Image.open(image_path)#.convert('L')
        #images.append(np.array(im)[...,:3])
        #im.close()

        features = np.array([np.hstack([self.__fd_histogram(image), self.__fd_haralick(image), self.__fd_hu_moments(image)]) for image in images])
        features = features.reshape((len(images), -1))

        # scaler = load(norm_path)
        return self.scaler.transform(features)

    def __threshold_pred(self, im):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 250

        params.filterByCircularity = True
        params.minCircularity = 0.05

        params.filterByConvexity = False

        params.filterByInertia = False
        # params.minInertiaRatio = 0.1
        detector = cv2.SimpleBlobDetector_create(params)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,230])
        upper_white = np.array([179,10,255])

        mask = cv2.inRange(img, lower_white, upper_white)
        keypoints = detector.detect(mask)

        centre_mask = 255 - np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
        if keypoints:
            print("Found blob")
            centre, r = keypoints[0].pt, int(keypoints[0].size/2)
            cv2.circle(centre_mask, (int(centre[0]), int(centre[1])), r, (0, 0, 0), thickness=-1)
            # mask = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('centre mask', centre_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow('original', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = cv2.bitwise_and(mask, mask, mask=centre_mask)
        cv2.imshow('hsv mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        white_pix = np.sum(mask > 0)
        print(white_pix)

        thresh_pred = ()
        if white_pix > 240:
            thresh_pred = (0.0, 1.0)
        elif white_pix < 110:
            thresh_pred = (1.0, 0.0)
        else:
            thresh_pred = (0.5, 0.5)
        return thresh_pred


    def classify_ball(self, im):
        features = self.__build_features(im)
        prediction = self.alg.predict_proba(features)

        print("gbm prob of stripes: " + str(prediction[0][1]))
        thresh_pred = self.__threshold_pred(im)

        return int(round(prediction[0][1] * 0.5 + thresh_pred[1] * 0.5))

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test_image", help="path to test image",
#                         type=str, default="testing_data/t003.png")
#     parser.add_argument("--norm_params", help="path to normlaization params file",
#                         type=str, default="ball_classification_norm_params.joblib")
#     parser.add_argument("--classifier", help="path to classifier",
#                         type=str, default="ball_gbm.joblib")
#     args = parser.parse_args()
#
#     features = build_features(args.test_image, args.norm_params)
#     alg = load(args.classifier)
#     prediction = alg.predict(features)
#
#     if prediction[0] == 0:
#         print('Solids')
#     elif prediction[0] == 1:
#         print('Stripes')
#     else:
#         print('No ball')
#
# if __name__ == '__main__':
#     main()
