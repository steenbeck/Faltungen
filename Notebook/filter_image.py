from scipy import signal as signal
import matplotlib.pyplot as pyplot
import read_image

default_file_name = "test.png"


### Show Images
def show_image(image):
	pyplot.imshow(image, cmap = "gray")
	pyplot.show()

def show_test_image():
	show_image(read_image.read_image(default_file_name))


### Filter Images
def filter_image(image_filters, image_file = default_file_name):
	image = read_image.read_image(image_file)
	filtered = image

	for f in image_filters:
		filtered = signal.convolve2d(image, f)

	return filtered

def filter_and_show_image(image_filters, image_file = default_file_name):
	show_image(filter_image(image_filters, image_file))

