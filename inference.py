"""
Author: Software Engineer Muhammet Bolat
Copyright (C): 2020 Muhammet Bolat
Licence: Public Domain
"""

import os
import glob
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical
import numpy as np
from pandas import DataFrame
from keras.models import load_model


def display_stats(y_test: np.array = None, pred: np.array = None, columns: list = None) -> None:
    """
    This method helps you to see how the result of predictions.
    It includes accuracy, recall, precision, f1-score and confusion matrix.
    :param y_test: real values.
    :param pred: predicted values.
    :param columns: labels as numeric or string in list.
    :return: Anything. It prints results.
    """
    print(f"### Result of the predictions using {len(y_test)} test data ###\n")
    y_test_class = from_categorical(y_test)
    pred = from_categorical(pred)
    print("Classification Report:\n")
    print(classification_report(y_test_class, pred, target_names=columns))
    print("\nConfusion Matrix:\n\n")
    df_cm = DataFrame(confusion_matrix(y_test_class, pred), index=columns, columns=columns)
    print(df_cm)
    print("\nAccuracy:", round(accuracy_score(y_test_class, pred), 5))


def from_categorical(array: np.ndarray = None) -> list:
    """
    This method inverse input from one-hot encoding data to ordinal.
    Example: [[0,0,0,1,0], [1,0,0,0,0]] => [3,0]
    :param array: numpy array.
    :return: list array.
    """
    return [x.index(max(x)) for x in array.tolist()]


def readData(fileName: str = None, imageSize: tuple = (200, 200)) -> tuple:
    """
    This method reads all images from the given filename.
    :param fileName: relative or absolute path.
    :param imageSize: it is needed to set training image size. Default is (200, 200)
    :return: return all images and labels as tuple object.
    """
    datas = list()
    labels = list()

    for listPath in glob.glob("{}*".format(fileName)):
        label = listPath.split('/')[-1]
        for imgPath in glob.glob(os.path.join(listPath, '*')):
            img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            img = cv2.resize(img, imageSize)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            datas.append(img)
            labels.append(label)

    return datas, labels


########################################################################################################################
# MODEL PARAMETERS
TEST_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/HAM_DATA/TEST/'
ANN_MODEL_PATH = 'muhammetbolat_model.h5'

IMAGE_SIZE = (200, 200)
label_to_id = {'Armut': 0, 'Cilek': 1, 'Elma_Kirmizi': 2, 'Elma_Yesil': 3, 'Mandalina': 4, 'Muz': 5, 'Portakal': 6}


########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == "__main__":
    print("Reading test images from the given files.")
    test_data, test_label = readData(TEST_PATH, IMAGE_SIZE)

    print("*** PRE-PROCESSING ***")
    test_data = np.array(test_data) / 255
    test_label = to_categorical(np.array([label_to_id[item] for item in test_label]))

    print("*** MODEL IS LOADING ***")
    model = load_model(ANN_MODEL_PATH)

    print("*** Images are predicting... ***")
    prediction = model.predict(test_data)
    display_stats(test_label, prediction, list(label_to_id.keys()))

    print("*** FINISH ***")

