from matplotlib import pyplot as plt
import numpy as np 
import math
import matplotlib.pyplot as plt


def load_data(dataloc):
	data = np.loadtxt(dataloc, unpack='true')
	data = np.transpose(data, (1, 0))
	return data	


def extract_feature(image):
	image = np.reshape(image, (28, 28))
	flip_image = np.flip(image, 1)
	diff = abs(image-flip_image)
	sys = -sum(sum(diff))/784

	flip_image_ud = np.flip(image, 0)
	diff_ud = abs(image - flip_image_ud)
	sys_ud = -sum(sum(diff_ud.transpose()))/784

	intense = sum(sum(image))/784

	return sys, sys_ud, intense


def load_features(dataloc):
	data = load_data(dataloc)
	n, _ = data.shape
	data_set = []
	for i in range(n):
		label = 1 if data[i, 0]==1 else -1
		image = data[i, 1:]
		sys, intense = extract_feature(image)
		data_set.append([label, sys, intense])
	return np.array(data_set)[:,1:], np.array(data_set)[:,0]


def image_to_feature(images, labels):

	if images.shape[0] != labels.shape[0]:
		print("error: the number of samples is different from the number of labels")
		return

	n = images.shape[0]

	data_set = []

	for i in range(n):
		image = images[i, :, :]
		label = labels[i]

		tmp_list = [label] + list(extract_feature(image))
		data_set.append(tmp_list)

	return np.array(data_set)[:, 1:], np.array(data_set)[:, 0]
