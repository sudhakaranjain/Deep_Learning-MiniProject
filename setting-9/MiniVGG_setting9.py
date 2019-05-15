from tensorflow.keras import Sequential, utils, optimizers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
import idx2numpy
import cv2 as cv
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

class MiniVGG():

	def __init__(self):
		self.network = Sequential()
		
		self.network.add(Conv2D(32, (3, 3), padding="same", input_shape=(28,28,1)))
		self.network.add(Activation("relu"))
		self.network.add(BatchNormalization())

		self.network.add(Conv2D(32, (3, 3), padding="same"))
		self.network.add(Activation("relu"))
		self.network.add(BatchNormalization())

		self.network.add(MaxPooling2D(pool_size=(2, 2)))
		self.network.add(Dropout(0.1))

		# second CONV => RELU => CONV => RELU => POOL layer set
		self.network.add(Conv2D(64, (3, 3), padding="same"))
		self.network.add(Activation("relu"))
		self.network.add(BatchNormalization())

		self.network.add(Conv2D(64, (3, 3), padding="same"))
		self.network.add(Activation("relu"))
		self.network.add(BatchNormalization())

		self.network.add(MaxPooling2D(pool_size=(2, 2)))
		self.network.add(Dropout(0.3))

		# first (and only) set of FC => RELU layers
		self.network.add(Flatten())
		self.network.add(Dense(512))
		self.network.add(Activation("relu"))
		self.network.add(BatchNormalization())
		self.network.add(Dropout(0.5))

		# softmax classifier
		self.network.add(Dense(10))
		self.network.add(Activation("softmax"))

	def extract_data(self, filepath, num_img):
		with gzip.open(filepath) as f:
			f.read(16)
			buf = f.read(28 * 28 * num_img)
			train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
			train_data = train_data.reshape(num_img, 28, 28, 1)
			return train_data

	def extract_labels(self, filepath, num_img):
		with gzip.open(filepath) as f:
			f.read(8)
			buf = f.read(num_img)
			labels= np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
			# labels = labels.reshape(num_img, 28, 28, 1)
			return labels


if __name__ == "__main__":

	train_data_path = '../fashion_data/train-images-idx3-ubyte.gz'
	test_data_path = '../fashion_data/t10k-images-idx3-ubyte.gz'
	train_label_path = '../fashion_data/train-labels-idx1-ubyte.gz'
	test_label_path = '../fashion_data/t10k-labels-idx1-ubyte.gz'
	
	vgg = MiniVGG()
  
	train_data = vgg.extract_data(train_data_path, 60000)
	train_labels = vgg.extract_labels(train_label_path, 60000)
	test_data = vgg.extract_data(test_data_path, 10000)
	test_labels = vgg.extract_labels(test_label_path, 10000)


# show image using cv
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

# show image using matplotlib
	# image = np.asarray(train_data[512]).squeeze()
	# plt.imshow(image)
	# plt.show()

	train_labels = utils.to_categorical(train_labels)
	test_labels = utils.to_categorical(test_labels)
	opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	vgg.network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	hist = vgg.network.fit(train_data, train_labels, batch_size = 170, validation_data=(test_data, test_labels), epochs=30)
	vgg.network.save('trained_cnn.h5')
	with open('trainHistoryDict', 'wb') as file:
		pickle.dump(hist.history, file)
