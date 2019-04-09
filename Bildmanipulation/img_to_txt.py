import read_image
import json
import sys

def save_image_to(image, file_name):
	with open(file_name, "w") as file:
		json.dump(image.tolist(), file)


if __name__ == '__main__':
	save_image_to(read_image.read_image(), sys.argv[1])