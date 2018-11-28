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
from sklearn import cross_validation, metrics
import tensorflow as tf
from tensorflow import keras

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
                     input_shape=(20, 20, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(keras.layers.Conv2D(16, (2, 2), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))#, kernel_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(50, activation='relu'))#, kernel_regularizer=keras.regularizers.l2(0.00001)))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.00005, beta_1=0.98),
              metrics=['accuracy'])
    model.fit(train_input, train_target,
          batch_size=64,
          epochs=100,
          verbose=1,
          validation_data=(test_input, test_target))

def threshold_model_fit(train_input, train_target):
    max_white_pix = 0
    for idx, label in enumerate(train_target):
        if label == 0:
            img = train_input[idx]
            n_white_pix = np.sum(img > 230)
            if n_white_pix > 0:
                max_white_pix = max(max_white_pix, n_white_pix)
    print(max_white_pix)

def modelfit(alg, train_input, train_target, val_input, val_target, performCV=True, printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(train_input, train_target)

    #Predict training set:
    dtrain_predictions = alg.predict(train_input)
    # dtrain_predprob = alg.predict_proba(train_input)[:,1]

    val_predictions = alg.predict(val_input)
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
        im_path = 'train_augmented/' + str_label + '.png'
        im = Image.open(im_path)#.convert('L')
        images.append(np.array(im)[...,:3])
        im.close()

    targets = np.array([int(label[1]) for label in labels])
    # targets = np.eye(3)[targets]
    global_features = np.array([np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)]) for image in images])
    global_features = global_features.reshape((len(images), -1))
    train_input, train_target, test_input, test_target = split_train_test(global_features, targets)

    scaler = StandardScaler()
    train_input = scaler.fit_transform(train_input)
    test_input = scaler.transform(test_input)

    #train_input = train_input.reshape((train_input.shape[0], -1))
    #test_input = test_input.reshape((test_input.shape[0], -1))

    gbm0 = GradientBoostingClassifier(n_estimators = 50, max_depth = 3, random_state=10)
    # svm_clf = SVC(kernel='poly', degree=4, gamma='auto')
    modelfit(gbm0, train_input, train_target, test_input, test_target)
    # threshold_model_fit(train_input, train_target)
    # cnn_model(train_input, train_target, test_input, test_target)
    dump(scaler, 'ball_classification_norm_params.joblib')
    dump(gbm0, 'ball_classification_gbm.joblib')

    print(train_input.shape)
    print(test_input.shape)

if __name__ == '__main__':
    main()
