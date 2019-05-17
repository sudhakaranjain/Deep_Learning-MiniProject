import tensorflow.keras as keras
import idx2numpy
import cv2 as cv
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

class ResNet():

	def resnet_layer(self, inputs,
				 num_filters=16,
				 kernel_size=3,
				 strides=1,
				 activation='relu',
				 batch_normalization=True,
				 conv_first=True):

		conv = keras.layers.Conv2D(num_filters,
					  kernel_size=kernel_size,
					  strides=strides,
					  padding='same',
					  kernel_initializer='he_normal',
					  kernel_regularizer=keras.regularizers.l2(1e-4))

		x = inputs
		if conv_first:
			x = conv(x)
			if batch_normalization:
				x = keras.layers.BatchNormalization()(x)
			if activation is not None:
				x = keras.layers.Activation(activation)(x)
		else:
			if batch_normalization:
				x = keras.layers.BatchNormalization()(x)
			if activation is not None:
				x = keras.layers.Activation(activation)(x)
			x = conv(x)
		return x


	def resnet_v1(self, input_shape, depth=2, num_classes=10):
	
		if (depth - 2) % 6 != 0:
			raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
		# Start model definition.
		num_filters = 16
		num_res_blocks = int((depth - 2) / 6)

		inputs = keras.layers.Input(shape=input_shape)
		x = self.resnet_layer(inputs=inputs)
		# Instantiate the stack of residual units
		for stack in range(3):
			for res_block in range(num_res_blocks):
				strides = 1
				if stack > 0 and res_block == 0:  # first layer but not first stack
					strides = 2  # downsample
				y = self.resnet_layer(inputs=x,
								 num_filters=num_filters,
								 strides=strides)
				y = self.resnet_layer(inputs=y,
								 num_filters=num_filters,
								 activation=None)
				if stack > 0 and res_block == 0:  # first layer but not first stack
					# linear projection residual shortcut connection to match
					# changed dims
					x = self.resnet_layer(inputs=x,
									 num_filters=num_filters,
									 kernel_size=1,
									 strides=strides,
									 activation=None,
									 batch_normalization=False)
				x = keras.layers.add([x, y])
				x = keras.layers.Activation('relu')(x)
			num_filters *= 2

		# Add classifier on top.
		# v1 does not use BN after last shortcut connection-ReLU
		x = keras.layers.AveragePooling2D(pool_size=8)(x)
		y = keras.layers.Flatten()(x)
		outputs = keras.layers.Dense(num_classes,
						activation='softmax',
						kernel_initializer='he_normal')(y)

		# Instantiate model.
		model = keras.models.Model(inputs=inputs, outputs=outputs)
		return model

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
	
	res = ResNet()
  
	train_data = res.extract_data(train_data_path, 60000)
	train_labels = res.extract_labels(train_label_path, 60000)
	test_data = res.extract_data(test_data_path, 10000)
	test_labels = res.extract_labels(test_label_path, 10000)
	network = res.resnet_v1(input_shape=train_data.shape[1:])

# show image using cv
	# cv.imshow("", train_data[512])
	# cv.waitKey(0)

# show image using matplotlib
	# image = np.asarray(train_data[512]).squeeze()
	# plt.imshow(image)
	# plt.show()

	train_labels = keras.utils.to_categorical(train_labels)
	test_labels = keras.utils.to_categorical(test_labels)
	opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	hist = network.fit(train_data, train_labels, batch_size = 170, validation_data=(test_data, test_labels), epochs=110)
	network.save('trained_cnn.h5')
	with open('trainHistoryDict', 'wb') as file:
		pickle.dump(hist.history, file)
