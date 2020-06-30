from numpy import loadtxt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as k
import glob as glob
import os
import matplotlib.pyplot as plt


def NNanalysis(path, dataset, Xmax, labelCol, classNum):
    # one-hot encode label column
    #pd.get_dummies(dataset['quality'], prefix='quality')

    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = dataset.iloc[:, 0:Xmax]
    y = dataset.iloc[:, labelCol]
    # split data with 0.33 test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print("Now onto the ML code")
    # define the keras model
    model = Sequential()
    model.add(Dense(classNum, input_dim=Xmax, activation='relu'))
    model.add(Dense(classNum, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(classNum, activation='relu'))
    model.add(Dense(classNum, activation='softmax'))

    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)

    # compile the keras model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    bs = 10
    # fit the keras model on the dataset
    hist = model.fit(X, y, epochs=2000, batch_size=bs,
                     validation_data=(X_test, y_test))

    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    ACCf = path + "\ACCxEPOCH.png"
    LOSSf = path + "\LOSSxEPOCH.png"
    # summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
