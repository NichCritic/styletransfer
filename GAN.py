import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D
from keras.optimizers import SGD
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
import cv2
import os

G = Sequential()

G.add(VGG19(include_top=False, weights='imagenet'))
G.add(Conv2D(512, (3, 3), activation='relu'))
G.add(Conv2D(256, (3, 3), activation='relu'))
G.add(Conv2D(128, (3, 3), activation='relu'))
G.add(Conv2D(64, (3, 3), activation='relu'))

D = Sequential()
D.add(VGG19(include_top=False, weights='imagenet'))
D.add(Dense(4096, activation='relu'))
D.add(Dense(4096, activation='relu'))
D.add(Dense(2, activation='sigmoid'))

Gsgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
G.compile(loss='binary_crossentropy', optimizer=Gsgd, metrics=['accuracy'])


Dsgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
D.compile(loss='binary_crossentropy', optimizer=Dsgd, metrics=['accuracy'])


def generateFrames():
	cap = cv2.VideoCapture("H:\\Projects\\styletransfer\\styletransfer\\data\\atttackontitan.avi")

	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")

	while cap.isOpened():
		ret, frame = cap.read()
		yield frame	
		
		if not ret:
			break

	cap.release()
	cv2.destroyAllWindows()

def generateImage(G, source):
	sImage = load_img(path=source)
	sImageArray = img_to_array(sImage)
	sImageArray = K.variable(preprocess_input(np.expand_dims(sImageArray, axis=0)), dtype='float32')
	generated_frame = G.predict(sImageArray)
	return generated_frame






