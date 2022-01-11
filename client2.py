import os

import flwr as fl
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# define cnn model
def define_model():
    
	model = VGG19(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def create_model():
  model = Sequential()
  # Adds a densely-connected layer with 64 units to the model:
  model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))
  model.add(MaxPooling2D(pool_size = (2,2)))
  # Add another:
  model.add(Conv2D(64,(3,3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))

  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  # Add a softmax layer with 10 output units:
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer="adam",
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)

def create_test1_data(path):
    for p in os.listdir(path):
        id_line.append(p.split(".")[0])
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X_test.append(new_img_array)

if __name__ == "__main__":
    main_dir = "./client1/"
    train_dir = "train"
    path = os.path.join(main_dir,train_dir)
    X = []
    y = []
    convert = lambda category : int(category == 'dog')
    print(path)
    create_test_data(path)
    X = np.array(X).reshape(-1, 80,80,1)
    X = X/255.0 # Normalize data
    y = np.array(y)

    # Configuration
    train_dir = "test"
    path2 = os.path.join(main_dir,train_dir)

    X_test = []
    id_line = []

    create_test1_data(path2)
    X_test = np.array(X_test).reshape(-1,80,80,1)
    X_test = X_test/255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                    random_state=101)

    model = create_model()
    # Define Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(X_train, y_train, epochs=3, batch_size=10)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=Client())