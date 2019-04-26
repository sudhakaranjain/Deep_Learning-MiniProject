import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

with open('trainHistoryDict', 'rb') as file:
	a = pickle.load(file)

plt.plot(a['acc'])
plt.plot(a['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Apoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
