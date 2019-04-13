import imageio

def read_image(file_name):
	return imageio.imread(file_name, as_gray =  True, pilmode = "RGB")