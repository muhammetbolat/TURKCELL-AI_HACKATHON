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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
from keras.optimizers import Adamax
from keras.regularizers import l1
from pandas import DataFrame


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


def modelConfiguration() -> Model:
    """
    This method helps engineers to have readable code. It includes ANN/CNN configuration.
    :return: model object.
    """
    model = Sequential(name='Fruit Classification')
    model.add(Conv2D(16, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01),
                     input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), kernel_regularizer=l1(0.01), bias_regularizer=l1(0.01), padding='same'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adamax(),
                  metrics=['accuracy'])

    model.summary()

    return model


def augmentation() -> ImageDataGenerator:
    """
    This methods creates new images from avaiable images.
    :return: ImageDataGenerator object.
    """
    return ImageDataGenerator(rotation_range=20,
                              horizontal_flip=True,
                              vertical_flip=True,
                              shear_range=0.15,
                              width_shift_range=.3,
                              height_shift_range=.3,
                              zoom_range=0.15,
                              fill_mode="nearest")


########################################################################################################################
# MODEL PARAMETERS
TRAINING_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/HAM_DATA/TRAINING/'
VALIDATION_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/HAM_DATA/VALIDATION/'
TEST_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/HAM_DATA/TEST/'

MODEL_SAVE_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/muhammetbolat.h5'

IMAGE_SIZE = (200, 200)
EPOCHS = 24
BATCH_SIZE = 8

########################################################################################################################
########################################################################################################################
########################################################################################################################
if __name__ == "__main__":
    print("Reading images from the given files.")
    training_data, training_label = readData(TRAINING_PATH, IMAGE_SIZE)
    validation_data, validation_label = readData(VALIDATION_PATH, IMAGE_SIZE)
    test_data, test_label = readData(TEST_PATH, IMAGE_SIZE)

    print("*** PRE-PROCESSING ***")
    id_to_label = {key: value for key, value in enumerate(np.unique(training_label))}
    label_to_id = {key: value for value, key in id_to_label.items()}

    print("Normalize input data to 0-1 range")
    training_data = np.array(training_data) / 255
    validation_data = np.array(validation_data) / 255
    test_data = np.array(test_data) / 255

    print("labels are assigned to numeric value and encoded to one-hot.")
    training_label = to_categorical(np.array([label_to_id[item] for item in training_label]))
    validation_label = to_categorical(np.array([label_to_id[item] for item in validation_label]))
    test_label = to_categorical(np.array([label_to_id[item] for item in test_label]))

    print("Training: data shape = {a}, label shape =  {b}".format(a=training_data.shape, b=training_label.shape))
    print("Validation: data shape = {a}, label shape =  {b}".format(a=validation_data.shape, b=validation_label.shape))
    print("Test data shape = {a}, training labels shape =  {b}".format(a=test_data.shape, b=test_label.shape))

    print("Model configuration is called.")
    model = modelConfiguration()

    print("In order to prevent over fitting, all images are augmented.")
    image_augmentation = augmentation()

    print("**************************** TRAINING ****************************")
    history = model.fit_generator(image_augmentation.flow(training_data, training_label, batch_size=BATCH_SIZE),
                                  validation_data=(validation_data, validation_label),
                                  steps_per_epoch=len(training_data) // BATCH_SIZE,
                                  epochs=EPOCHS)

    print("****************************  TESTING  ****************************")
    prediction_result = model.predict(test_data)
    display_stats(test_label, prediction_result, list(label_to_id.keys()))

    model.save(MODEL_SAVE_PATH)

    print("****************************  FINISH  ****************************")
