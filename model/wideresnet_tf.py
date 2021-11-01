import tensorflow as tf
import tensorflow.keras.layers as tfkl


class Wide_Basic(tfkl.Layer):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        """

        :param in_planes: number of features
        :param planes:
        :param dropout_rate:
        :param stride:
        """
        super(Wide_Basic, self).__init__()
        self.bn1 = tfkl.BatchNormalization()
        # filters is the dim of the output space
        self.conv1 = tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, padding="same")
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bn2 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, padding="same", strides=stride)
        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = tf.keras.Sequential(
                tfkl.Conv2D(filters=out_planes, kernel_size=1, use_bias=True,  strides=stride)
            )

    def call(self, x):
        out = self.conv1(tf.nn.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
