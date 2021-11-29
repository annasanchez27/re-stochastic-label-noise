import tensorflow as tf
import numpy as np
import argparse
import yaml
import wandb

from wandb.keras import WandbCallback
from pathlib import Path

from dataset import get_dataset
from model.wrn import WideResNet
from model.utils import CheckpointSaver


def main():
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    parser = argparse.ArgumentParser(description="re-SLN")
    parser.add_argument(
        "--dataset", type=str, help="Dataset to use.", default=config["dataset"]
    )
    parser.add_argument(
        "--noise_mode", type=str, help="Noise mode.", default=config["noise_mode"]
    )
    parser.add_argument(
        "--noise_rate", type=float, help="Noise rate.", default=config["noise_rate"]
    )
    parser.add_argument(
        "--sigma", type=float, help="Sigma parameter for SLN.", default=0.0
    )
    parser.add_argument(
        "--use_sln", type=bool, help="Specify if SLN should be used", default=False
    )
    parser.add_argument(
        "--perform_theta_experiment", type=bool, help="Specify if theta experiment is run", default=False
    )
    parser.add_argument(
        "--perform_sln_experiment", type=bool, help="Specify if sln experiment is run", default=False
    )
    parser.add_argument(
        "--sln_mode", type=str, help="SLN mode for adding noise.", default=config["sln_mode"]
    )
    args = parser.parse_args()

    dataset = args.dataset
    mean = config[dataset]["mean"]
    variance = np.square(config[dataset]["std"])
    batch_size = config["batch_size"]
    sigma = config["sigma"]

    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100

    if args.use_sln:
        # On CIFAR-10, we use σ = 1 for symmetric noise and σ = 0.5 otherwise; On CIFAR-100, we
        # use σ = 0.1 for instance-dependent noise and σ = 0.2 otherwise.
        if dataset == "cifar10":
            sigma = 1.0 if args.noise_mode == "symmetric" else 0.5
        if dataset == "cifar100":
            sigma = 0.1 if args.noise_mode == "instance_dependent" else 0.2

    if args.perform_theta_experiment:
        wandb_project = "re-stochastic-label-noise-theta_experiment"
        sigma = args.sigma
    elif args.perform_sln_experiment:
        wandb_project = "re-stochastic-label-noise-sln_experiment"
    else:
        wandb_project = "re-stochastic-label-noise"

    wandb.init(project=wandb_project, entity="sebastiaan")

    wandb.config.update({
        "dataset": dataset,
        "noise_rate": args.noise_rate,
        "noise_mode": args.noise_mode,
        "use_sln": args.use_sln,
        "sigma": sigma,
        "batch_size": batch_size,
        "learning_rate": config["learning_rate"],
        "momentum": config["momentum"],
        "weight_decay": config["weight_decay"],
        "epochs": config["epochs"],
        "accumulation_steps": config["accumulation_steps"],
        "effective_batch_size": config["accumulation_steps"] * batch_size,
        "sln_mode": args.sln_mode,
    })

    print(
        f"Training model on dataset: {dataset}, with noise mode: {args.noise_mode}, with noise rate: {args.noise_rate} and sigma: {sigma}")

    (train_images, train_labels), (test_images, test_labels) = get_dataset(
        dataset,
        noise_mode=args.noise_mode,
        noise_rate=args.noise_rate,
        path=config["path"],
        batch_size=batch_size,
    )

    saver = CheckpointSaver(
        k=config["save_every_kth_epoch"], checkpoint_path=config["checkpoint_path"]
    )
    # https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
    input_shape = (None, 32, 32, 3)
    model = WideResNet(mean, variance, sigma, ga_steps=config["accumulation_steps"], inputs=input_shape, sln_mode=args.sln_mode, num_classes=num_classes)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(
        momentum=config["momentum"], learning_rate=config["learning_rate"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        validation_freq=2,
        callbacks=[saver, WandbCallback(monitor="train_loss")],
        epochs=config["epochs"],
        batch_size=batch_size,
        shuffle=True,
    )
    model.save_weights(
        f"{config['checkpoint_path']}/final_model_{dataset}_{args.noise_mode}_{args.noise_rate}_{sigma}",
        save_format="h5",
    )


if __name__ == "__main__":
    main()
