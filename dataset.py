import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split

def get_dataset(dataset, noise_mode, noise_rate, path):
    if dataset == "cifar10":
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = datasets.cifar10.load_data()
    elif dataset == "cifar100":
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = datasets.cifar100.load_data()
    else:
        raise ValueError(f"incorrect dataset provided: {dataset}")

    # normalize between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # thats the only way to imitate the randomcrop effect that pytorch does, cause tf does not do padding
    train_images = tf.image.resize_with_crop_or_pad(train_images, 40, 40).numpy()

    if noise_mode == "openset" and dataset == "cifar10":
        # replace part of CIFAR-10 images with CIFAR-100 images as done in the original code
        (cifar100, _), _ = datasets.cifar100.load_data()

        index1 = np.random.choice(
            len(train_images), int(len(train_images) * noise_rate), replace=False
        )
        index2 = np.random.choice(
            len(train_images), int(len(train_images) * noise_rate), replace=False
        )

        train_images[index1] = cifar100[index2]

    elif noise_mode in {"symmetric", "asymmetric", "instance_dependent"}:
        # read noisy labels and change the train_labels
        labels_path = f"{path}/{dataset}/{noise_mode}/labels_{noise_rate}.npy"
        train_labels = np.load(labels_path)
    elif noise_mode == "original":
        # keep correct labels
        train_labels = np.squeeze(train_labels)
    else:
        raise ValueError(f"Incorrect noise_mode provided: {noise_mode}")

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1,
                                                                          random_state=42)

    train_labels = tf.one_hot(train_labels, 10)
    val_labels = tf.one_hot(val_labels, 10)
    test_labels = tf.one_hot(test_labels, 10)

    return (train_images[:500, :, :, :], train_labels[:500, :]), (val_images[:500, :, :, :], val_labels[:500, :]), (test_images, test_labels)