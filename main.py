from pathlib import Path
import numpy as np
import yaml
import wandb


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

    (train_images, train_labels), (test_images, test_labels) = get_dataset(
        dataset,
        noise_mode=config["noise_mode"],
        noise_rate=config["noise_rate"],
        path=config["path"],
    )

    model = WideResNet(mean, variance)
    saver = CheckpointSaver(
        k=config["save_every_kth_epoch"], checkpoint_path=config["checkpoint_path"]
    )

    steps = train_images.shape[0] // batch_size
    model.compile(optimizer="adam")
    model.fit(
        train_images,
        train_labels,
        callbacks=[saver],
        epochs=config["epochs"],
        steps_per_epoch=steps,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
