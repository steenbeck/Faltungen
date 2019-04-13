
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

import os


def fashion_label(i):
	if i > 9 or i < 0:
		raise Exception("")
	return [
		"T-Shirt",
		"Hose",
		"Pullover",
		"Kleid",
		"Jacke",
		"Sandale",
		"Hemd",
		"Sneaker",
		"Tasche",
		"Stiefelette"
	][i]


def get_preprocessed_fashion_mnist():
	(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

	train_images = train_images.reshape((60000, 28, 28, 1))
	test_images = test_images.reshape((10000, 28, 28, 1))

	# normalize pixel values to be between 0 and 1
	train_images, test_images = train_images / 255, test_images / 255

	return (train_images, train_labels), (test_images, test_labels)


# Create Model and train it using the given data
def get_trained_fashion_mnist_model(train_images, train_labels):

	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))

	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))


	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs = 1)

	return model


# Show images
def show_selected_fashion(test_images, test_labels):
	fig = plt.figure(figsize=(20, 20))
	columns = 4
	rows = 5

	for i in range(1, columns*rows + 1):
		image = test_images[i]
		image = np.array(image, dtype='float')
		pixels = image.reshape((28, 28))

		subplt = fig.add_subplot(rows, columns, i)
		subplt.axis("off")

		plt.imshow(pixels, cmap='gray')

	plt.show()



def show_predicted_selected_fashion(test_images, test_labels):
	predicted = model.predict_classes(test_images)

	columns = 3
	rows = 3

	f, axarr = plt.subplots(rows, columns)

	for i in range(0, rows):
		for j in range(0, columns):
			index = i*columns + j
			image = test_images[index]

			image = np.array(image, dtype='float')
			pixels = image.reshape((28, 28))

			subplt = axarr[i, j]
			subplt.axis("off")

			lbl = "Wahrheit: " + fashion_label(test_labels[index]) + "\n CNN: " + fashion_label(predicted[index])
			subplt.set_title(lbl, fontsize = 12)

			subplt.imshow(pixels, cmap='gray')


	f.subplots_adjust(hspace = 0.5)
	plt.show()







### MAIN ###

if __name__ == "__main__":

	# log settings
	tf.logging.set_verbosity(tf.logging.ERROR)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


	# get data
	(train_images, train_labels), (test_images, test_labels) = get_preprocessed_fashion_mnist()

	# display selected fashion
	show_selected_fashion(test_images, test_labels)

	# train model
	model = get_trained_fashion_mnist_model(train_images, train_labels)


	# display predicted fashion
	show_predicted_selected_fashion(test_images, test_labels)
