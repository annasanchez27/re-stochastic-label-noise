import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np


def get_dataset(dataset, noise_mode, noise_rate, path):
	if dataset == 'cifar10':
		(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	elif dataset == 'cifar100':
		(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
	else:
		raise ValueError(f'incorrect dataset provided: {dataset}')

	# normalize between 0 and 1
	train_images, test_images = train_images / 255.0, test_images / 255.0

	# thats the only way to imitate the randomcrop effect that pytorch does, cause tf does not do padding
	# maybe this is better just before the random crop in the model and do it just for that batch (but done each time)
	train_images = tf.image.resize_with_crop_or_pad(train_images, 40, 40)
	# train_images = np.pad(train_images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant')

	# randomcrop, randomflip and mean std normalization is recommended to make it inside the model as a layer.
	# randomcrop and randomflip are only performed in training mode, not inference mode.

	if noise_mode == 'openset' and dataset == 'cifar10':
		# replace part of CIFAR-10 images with CIFAR-100 images as done in the original code
		(cifar100, _), _ = datasets.cifar100.load_data()

		index1 = np.random.choice(len(train_images), int(len(train_images) * noise_rate), replace=False)
		index2 = np.random.choice(len(train_images), int(len(train_images) * noise_rate), replace=False)

		train_images[index1] = cifar100[index2]

	# TODO: change the files of the instance_dependent to labels like the others
	elif noise_mode in {'symmetric', 'asymmetric', 'instance_dependent'}:
		# read noisy labels and change the train_labels
		labels_path = f'{path}/{dataset}/{noise_mode}/labels_{noise_rate}.npy'
		train_labels = np.load(labels_path).reshape(-1, 1)
	elif noise_mode == 'original':
		# keep correct labels
		pass
	else:
		raise ValueError(f'Incorrect noise_mode provided: {noise_mode}')

	# also return normalization values or keeping in config file
	return (train_images, train_labels), (test_images, test_labels)