import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import yaml
import wandb

from wandb.keras import WandbCallback
from pathlib import Path
from tqdm import tqdm

from dataset import get_dataset
from model.wrn import WideResNet
from model.utils import CheckpointSaver

wandb.init(project="re-stochastic-label-noise", entity="sebastiaan")


def main():
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
    wandb.config = config

    dataset = config["dataset"]
    mean = config[dataset]["mean"]
    variance = np.square(config[dataset]["std"])
    batch_size = config["batch_size"]
    sigma = config["sigma"]

    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = get_dataset(
        dataset,
        noise_mode=config["noise_mode"],
        noise_rate=config["noise_rate"],
        path=config["path"],
        batch_size=batch_size,
    )

    saver = CheckpointSaver(
        k=config["save_every_kth_epoch"], checkpoint_path=config["checkpoint_path"]
    )

    model = WideResNet(mean, variance, sigma)
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tfa.optimizers.SGDW(
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
        learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss=loss)


    model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        validation_freq=10,
        callbacks=[saver, WandbCallback(monitor="train_loss")],
        epochs=config["epochs"],
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
