import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

class Adam():

	def __init__(self):
	 self.network = keras.Sequential()
	 self.network.add(keras.layers.Conv2D(32, kernel_size=3, kernel_regularizer=keras.regularizers.l2(0.01), 
									  bias_regularizer=keras.regularizers.l2(0.01), 
									  activation='sigmoid', input_shape=(28,28,1)))
	 self.network.add(keras.layers.BatchNormalization())
	 self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	 self.network.add(keras.layers.Dropout(0.1))
	 self.network.add(keras.layers.Conv2D(64, kernel_size=3, kernel_regularizer=keras.regularizers.l2(0.01), 
										  bias_regularizer=keras.regularizers.l2(0.01), activation='sigmoid'))
	 self.network.add(keras.layers.BatchNormalization())
	 self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	 self.network.add(keras.layers.Dropout(0.3))
	 self.network.add(keras.layers.Flatten())
	 self.network.add(keras.layers.Dense(10, activation='softmax'))

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

	adam = Adam()

	train_data = adam.extract_data(train_data_path, 60000)
	train_labels = adam.extract_labels(train_label_path, 60000)
	test_data = adam.extract_data(test_data_path, 10000)
	test_labels = adam.extract_labels(test_label_path, 10000)

# show image using cv
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

# show image using matplotlib
	# image = np.asarray(train_data[512]).squeeze()
	# plt.imshow(image)
	# plt.show()

	datagen = keras.preprocessing.image.ImageDataGenerator(
  	rotation_range=10,
  	width_shift_range=0.05,
  	height_shift_range=0.05)

	datagen.fit(train_data)

# show augmented image using matplotlib
	# xdata, xlabel = datagen.flow(train_data, train_labels, batch_size=10).next()
	# print(xlabel[0])
	# image = np.asarray(xdata[0]).squeeze()
	# plt.imshow(image)
	# plt.show()

	train_labels = keras.utils.to_categorical(train_labels)
	test_labels = keras.utils.to_categorical(test_labels)
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	adam.network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	hist = adam.network.fit_generator(datagen.flow(train_data, train_labels, batch_size=170), validation_data=(test_data, test_labels), epochs=200)
	adam.network.save('trained_cnn.h5')
	with open('trainHistoryDict', 'wb') as file:
		pickle.dump(hist.history, file)
	