import tensorflow as tf
import numpy as np
import wandb
import tqdm
import os


class CheckpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, k: int, checkpoint_path: str) -> None:
        super(CheckpointSaver, self).__init__()
        self.k = k
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.k == 0:
            self.model.save_weights(f"{self.checkpoint_path}/model_checkpoint_{epoch}", save_format="h5", overwrite=True)

            if os.path.exists(
                f"{self.checkpoint_path}/model_checkpoint_{epoch}"
            ) and os.path.exists(
                f"{self.checkpoint_path}/model_checkpoint_{epoch - self.k}"
            ):
                os.remove(
                    f"{self.checkpoint_path}/model_checkpoint_{epoch - self.k}"
                )


def train(
    train_images, train_labels, val_images, val_labels, model, batch_size, epochs=10
):

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # Iterate over the batches of the dataset
        for step, batch_data in tqdm(enumerate(train_dataset)):
            loss_value = model.train_step(batch_data)
            train_loss.append(float(loss_value["loss"]))

        if epoch % 10:
            for step, (x_batch_val, y_batch_val) in tqdm(enumerate(val_dataset)):
                val_loss_value = model.train_step(x_batch_val, y_batch_val, train=False)
                val_loss.append(float(val_loss_value["loss"]))

        wandb.log(
            {
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "val_loss": np.mean(val_loss) if val_loss else None,
            }
        )
