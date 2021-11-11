import tensorflow as tf


class CheckpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, k: int, checkpoint_path: str) -> None:
        super(CheckpointSaver, self).__init__()
        self.k = k
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.k == 0:
            self.model.save(f"{self.checkpoint_path}/model_checkpoint_{epoch}.hd5")
