import os
import sys
import math
import cv2
from PIL import Image
import numpy as np
import mahotas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load
# from sklearn import cross_validation
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0).flatten()
    return haralick

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins = 32
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def cnn_model(train_input, train_target, test_input, test_target, num_classes = 3):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(8, kernel_size=(2, 2), strides=(1, 1),
                     activation='relu',
                     padding='same',
                     input_shape=(34, 34, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00005, beta_1=0.98),
              metrics=['accuracy'])
    model.fit(train_input, train_target,
          batch_size=64,
          epochs=100,
          verbose=1,
          validation_data=(test_input, test_target))

def threshold_predict(train_input):
    preds = []
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

    for idx in range(len(train_input)):
        img = cv2.cvtColor(train_input[idx], cv2.COLOR_BGR2HSV)
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

        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = cv2.bitwise_and(mask, mask, mask=centre_mask)
        cv2.imshow('hsv mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        white_pix = np.sum(mask > 0)
        print(white_pix)

        if white_pix > 290:
            preds.append((0.0, 1.0))
        elif white_pix < 100:
            preds.append((1.0, 0.0))
        else:
            preds.append((0.5, 0.5))

    return preds

def threshold_model_fit(train_input, train_target):
    max_white_pix = 0
    predictions = []

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 700
    params.maxArea = 1200

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = False

    params.filterByInertia = False
    # params.minInertiaRatio = 0.1
    detector = cv2.SimpleBlobDetector_create(params)

    for idx, label in enumerate(train_target):
        if label == 2:
            mask = cv2.cvtColor(train_input[idx], cv2.COLOR_BGR2HSV)

            # img = cv2.bitwise_and(img, img, black_mask)
            # lower_white = np.array([0,0,230])
            # upper_white = np.array([179,10,255])
            # mask = cv2.inRange(img, lower_white, upper_white)

            keypoints = detector.detect(mask)
            if not (keypoints == None or keypoints == []):
                print("Found blob")
                mask = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            predictions.append(mask)
            # continue

            cv2.imshow('hsv mask', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('original', train_input[idx])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            num_white_pix = np.sum(mask > 0)
            print(num_white_pix)
    hist = cv2.calcHist(predictions,[0],None,[256],[0,256])
    plt.plot(hist,color = 'r')
    plt.xlim([0,256])
    #hist = cv2.calcHist(predictions,[2],None,[256],[0,256])
    #plt.plot(hist,color = 'b')
    #plt.xlim([0,256])
    plt.show()
    # break
    print(max_white_pix)

def modelfit(alg, train_input, train_target, val_input, val_target, additionalPreds=None, performCV=False, printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(train_input, train_target)

    #Predict training set:
    dtrain_predictions = alg.predict(train_input)
    # dtrain_predprob = alg.predict_proba(train_input)[:,1]

    probs = alg.predict_proba(val_input)
    print(probs)
    val_predictions = alg.predict(val_input)

    if additionalPreds:
        val_predictions = []
        for idx, prob in enumerate(probs):
            val_predictions.append(int(round(0.5*prob[1] + 0.5*additionalPreds[idx][1])))
        val_predictions = np.array(val_predictions)
    #Perform cross-validation:
    #if performCV:
    #    cv_score = cross_validation.cross_val_score(alg, train_input, train_target, cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print("\nModel Report")
    print("Train Accuracy : %.4g" % metrics.accuracy_score(train_target, dtrain_predictions))
    print("Validation Accuracy : %.4g" % metrics.accuracy_score(val_target, val_predictions))
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(train_target, dtrain_predprob)

    #if performCV:
    #    print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

def split_train_test(input, target, train_percent=0.8):
    indices = list(range(input.shape[0]))
    np.random.shuffle(indices)
    num_train = int(input.shape[0]*train_percent)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices], input[test_indices,:], target[test_indices]

def main():
    np.random.seed(101)
    labels = np.load('train_augmented_labels.npy')
    images = []
    for str_label, label in labels:
        im_path = 'train_augmented/' + str_label.decode('UTF-8') + '.png'
        im = Image.open(im_path)#.convert('L')
        images.append(np.array(im)[...,:3])
        im.close()

    targets = np.array([int(label[1]) for label in labels])
    train_input, train_target, test_input, test_target = split_train_test(np.array(images), targets)
    threshold_model_fit(train_input, train_target)
    return
    threshold_preds = threshold_predict(test_input)
    train_input = np.array([np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)]) for image in train_input])
    test_input = np.array([np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)]) for image in test_input])
    # targets = np.eye(2)[targets]
    # global_features = np.array([np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)]) for image in images])
    # global_features = global_features.reshape((len(images), -1))

    scaler = StandardScaler()
    train_input = scaler.fit_transform(train_input)
    test_input = scaler.transform(test_input)

    #train_input = train_input.reshape((train_input.shape[0], -1))
    #test_input = test_input.reshape((test_input.shape[0], -1))

    gbm0 = GradientBoostingClassifier(n_estimators = 50, max_depth = 3, random_state=10)
    # svm_clf = SVC(kernel='poly', degree=4, gamma='auto')
    modelfit(gbm0, train_input, train_target, test_input, test_target, threshold_preds)
    # threshold_model_fit(train_input, train_target)
    # cnn_model(train_input, train_target, test_input, test_target, 2)
    dump(scaler, 'ball_classification_norm_params.joblib')
    dump(gbm0, 'ball_classification_gbm.joblib')

    print(train_input.shape)
    print(test_input.shape)

if __name__ == '__main__':
    main()
