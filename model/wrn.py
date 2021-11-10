import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.layers.experimental import preprocessing


def conv3x3(out_planes, stride=1):
    return tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, strides=stride, padding="same")


class WideBasic(tfkl.Layer):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        """
        :param in_planes: number of features
        :param planes:
        :param dropout_rate:
        :param stride:
        """
        super(WideBasic, self).__init__()
        self.bn1 = tfkl.BatchNormalization()
        self.conv1 = tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, padding="same")
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bn2 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, padding="same", strides=stride)
        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = tf.keras.Sequential(
                tfkl.Conv2D(filters=out_planes, kernel_size=1, use_bias=True, strides=stride)
            )

    def call(self, x):
        out = self.conv1(tf.nn.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(tfk.Model):
    def __init__(self, mean, variance, dropout_rate=0, depth=28, widen_factor=2, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        # normalization layer
        self.data_augmentation = tf.keras.Sequential(
            [preprocessing.RandomCrop(32, 32),
             preprocessing.RandomFlip("horizontal")])
        self.normalize = preprocessing.Normalization(mean=mean, variance=variance)
        self.conv1 = conv3x3(nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = tfkl.BatchNormalization(momentum=0.9)
        self.linear = tfkl.Dense(num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return tf.keras.Sequential(layers)

    def call(self, x, get_feat=False):
        out = self.data_augmentation(x)
        out = self.normalize(out)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = tf.nn.relu(self.bn1(out))
        out = tf.nn.avg_pool2d(out, 8, strides=None, padding="VALID")
        out = tf.reshape(out, (tf.shape(out)[0], -1))

        if get_feat:
            return out
        else:
            return self.linear(out)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = tfk.losses.categorical_crossentropy(y, logits)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}
