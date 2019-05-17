import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

with open('setting1', 'rb') as file:
	a1 = pickle.load(file)
with open('setting2', 'rb') as file:
	a2 = pickle.load(file)
with open('setting3', 'rb') as file:
	a3 = pickle.load(file)
with open('setting4', 'rb') as file:
	a4 = pickle.load(file)
with open('setting5', 'rb') as file:
	a5 = pickle.load(file)
with open('setting6', 'rb') as file:
	a6 = pickle.load(file)
with open('setting7', 'rb') as file:
	a7 = pickle.load(file)
with open('setting8', 'rb') as file:
	a8 = pickle.load(file)
with open('setting9', 'rb') as file:
	a9 = pickle.load(file)

# with open('trainHistoryDict_poor', 'rb') as file:
# 	a = pickle.load(file)

p1 = plt.subplot(231)
p1.plot(a2['val_acc'], 'r')
p1.plot(a3['val_acc'], 'g')
p1.plot(a4['val_acc'], 'k')
p1.set_title('(a)')
p1.set_ylabel('Accuracy')
p1.set_xlabel('Epoch')
p1.legend(['Setting-2', 'Setting-3', 'Setting-4'], loc='lower right')

p2 = plt.subplot(232)
p2.plot(a3['val_acc'], 'r')
p2.plot(a5['val_acc'], 'g')
p2.plot(a6['val_acc'], 'k')
p2.set_title('(b)')
p2.set_ylabel('Accuracy')
p2.set_xlabel('Epoch')
p2.legend(['Setting-3', 'Setting-5', 'Setting-6'], loc='lower right')

p3 = plt.subplot(233)
p3.plot(a1['val_acc'], 'r')
p3.plot(a2['val_acc'], 'g')
p3.set_title('(c)')
p3.set_ylabel('Accuracy')
p3.set_xlabel('Epoch')
p3.legend(['Setting-1', 'Setting-2'], loc='lower right')

p4 = plt.subplot(234)
p4.plot(a6['val_acc'], 'r')
p4.plot(a7['val_acc'], 'g')
p4.set_title('(d)')
p4.set_ylabel('Accuracy')
p4.set_xlabel('Epoch')
p4.legend(['Setting-6', 'Setting-7'], loc='lower right')

p5 = plt.subplot(235)
p5.plot(a4['val_acc'], 'r')
p5.plot(a8['val_acc'], 'g')
p5.plot(a9['val_acc'], 'k')
p5.set_title('(e)')
p5.set_ylabel('Accuracy')
p5.set_xlabel('Epoch')
p5.legend(['Setting-4', 'Setting-8', 'Setting-9'], loc='lower right')

plt.tight_layout()
plt.show()
