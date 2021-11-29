import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_dataset(dataset, noise_mode, noise_rate, path, batch_size):
    if dataset == "cifar10":
        num_classes = 10
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = datasets.cifar10.load_data()
    elif dataset == "cifar100":
        num_classes = 100
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = datasets.cifar100.load_data()
    else:
        raise ValueError(f"incorrect dataset provided: {dataset}")

    ground_truth_train_labels = np.array(train_labels)

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

        ground_truth_train_labels[index1] += 1
        train_labels = np.squeeze(train_labels)

    elif noise_mode in {"symmetric", "asymmetric", "instance_dependent"}:
        # read noisy labels and change the train_labels
        labels_path = f"{path}/{dataset}/{noise_mode}/labels_{noise_rate}.npy"
        train_labels = np.load(labels_path)
    elif noise_mode == "original":
        # keep correct labels
        train_labels = np.squeeze(train_labels)
    else:
        raise ValueError(f"Incorrect noise_mode provided: {noise_mode}")

    # normalize between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # only way to imitate the random crop effect that pytorch does, cause tf does not do padding
    train_images = tf.image.resize_with_crop_or_pad(train_images, 40, 40).numpy()

    print(f"accuracy: {accuracy_score(train_labels, ground_truth_train_labels)}")

    train_labels = np.column_stack(
        (
            tf.one_hot(train_labels, num_classes),
            tf.one_hot(np.squeeze(ground_truth_train_labels), num_classes),
        )
    )
    test_labels = tf.one_hot(np.squeeze(test_labels), num_classes)


    train_images, train_labels = make_divisible_by_batch(
        train_images, train_labels, batch_size
    )
    test_images, test_labels = make_divisible_by_batch(
        test_images, test_labels, batch_size
    )

    return (
        (train_images, train_labels),
        (test_images, test_labels),
    )


def make_divisible_by_batch(x, y, batch_size):
    k = x.shape[0] % batch_size
    return x[:-k, :, :, :], y[:-k, :]
