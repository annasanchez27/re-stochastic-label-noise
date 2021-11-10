import tensorflow as tf
import tensorflow.keras.layers as tfkl


def conv3x3(out_planes, stride=1):
    return tfkl.Conv2D(filters=out_planes, kernel_size=3, use_bias=True, strides=stride, padding="same")


"""def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
"""


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
                tfkl.Conv2D(filters=out_planes, kernel_size=1, use_bias=True, strides=stride)
            )

    def call(self, x):
        out = self.conv1(tf.nn.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(tfkl.Layer):
    def __init__(self, dropout_rate=0, depth=28, widen_factor=2, num_classes=10):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(nStages[0])
        self.layer1 = self._wide_layer(Wide_Basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(Wide_Basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(Wide_Basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = tfkl.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = tfkl.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return tf.keras.Sequential(*layers)
    def call(self, x, get_feat=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = tf.nn.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        if get_feat:
            return out
        else:
            return self.linear(out)