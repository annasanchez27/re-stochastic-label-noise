from pathlib import Path
import numpy as np
import yaml

from dataset import get_dataset
from model.wrn import WideResNet


def main():
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    dataset = config["dataset"]
    mean = config[dataset]["mean"]
    variance = np.square(config[dataset]["std"])
    batch_size = config["batch_size"]
    sigma = config["sigma"]

    (train_images, train_labels), (test_images, test_labels) = get_dataset(
        dataset,
        noise_mode=config["noise_mode"],
        noise_rate=config["noise_rate"],
        path=config["path"],
    )

    model = WideResNet(mean, variance, sigma)

    steps = train_images.shape[0] // batch_size
    model.fit(
        train_images,
        train_labels,
        epochs=config["epochs"],
        steps_per_epoch=steps,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
