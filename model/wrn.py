import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.keras.layers.experimental import preprocessing


def conv3x3(out_planes, stride=1):
    return tfkl.Conv2D(
        filters=out_planes, kernel_size=3, use_bias=True, strides=stride, padding="same"
    )


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
        self.conv1 = tfkl.Conv2D(
            filters=out_planes, kernel_size=3, use_bias=True, padding="same"
        )
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bn2 = tfkl.BatchNormalization()
        self.conv2 = tfkl.Conv2D(
            filters=out_planes,
            kernel_size=3,
            use_bias=True,
            padding="same",
            strides=stride,
        )
        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = tf.keras.Sequential(
                tfkl.Conv2D(
                    filters=out_planes, kernel_size=1, use_bias=True, strides=stride
                )
            )

    def call(self, x):
        out = self.conv1(tf.nn.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(tfk.Model):
    def __init__(
            self, mean, variance, sigma, ga_steps, inputs, sln_mode, dropout_rate=0, depth=28,
            widen_factor=2, num_classes=10, *args, **kwargs
    ):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.sigma = sigma
        self.num_classes = num_classes

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        # normalization layer
        self.data_augmentation = tf.keras.Sequential(
            [preprocessing.RandomCrop(32, 32), preprocessing.RandomFlip("horizontal")]
        )
        self.normalize = preprocessing.Normalization(mean=mean, variance=variance)
        self.conv1 = conv3x3(nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = tfkl.BatchNormalization(momentum=0.9)
        self.linear = tfkl.Dense(num_classes)
        self.flatten = tfkl.Flatten()

        self.cat_accuracy = tfk.metrics.CategoricalAccuracy()

        self.build(inputs)

        # gradient accumulation
        self.n_gradients = tf.constant(ga_steps, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
            self.trainable_variables]

        self.sln_mode = sln_mode

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
        out = self.flatten(out)

        if get_feat:
            return out
        else:
            return self.linear(out)

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, labels = data
        y = labels[:, :self.num_classes]
        ground_truth = tf.cast(tf.math.argmax(labels[:, self.num_classes:], axis=1), tf.float32)
        labels_idx = tf.math.argmax(y, axis=1)
        labels_idx = tf.cast(labels_idx, tf.float32)

        noisy_x = tf.gather_nd(x, tf.where(ground_truth != labels_idx))
        noisy_y = tf.gather_nd(y, tf.where(ground_truth != labels_idx))
        clean_x = tf.gather_nd(x, tf.where(ground_truth == labels_idx))
        clean_y = tf.gather_nd(y, tf.where(ground_truth == labels_idx))

        logits_noisy_y = self(noisy_x, training=False)
        logits_clean_y = self(clean_x, training=False)
        clean_loss = self.compiled_loss(clean_y, logits_clean_y)
        noisy_loss = self.compiled_loss(noisy_y, logits_noisy_y)

        if self.sigma > 0:
            if self.sln_mode == "both":
                y += self.sigma * tf.random.normal([y.shape[1]])
            if self.sln_mode == "clean":
                # Only apply noise to clean samples
                clean_y += self.sigma * tf.random.normal([clean_y.shape[1]])
                y = tf.concat([clean_y, noisy_y], axis=0)
                x = tf.concat([clean_x, noisy_x], axis=0)
            if self.sln_mode == "noisy":
                # Only apply noise to noisy samples
                noisy_y += self.sigma * tf.random.normal([noisy_y.shape[1]])
                y = tf.concat([clean_y, noisy_y], axis=0)
                x = tf.concat([clean_x, noisy_x], axis=0)

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.compiled_loss(y, logits)

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables
                               if 'bias' not in v.name]) * 0.0005

            loss += lossL2

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])

        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients,
                lambda: None)

        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss, "clean_loss": clean_loss, "noisy_loss": noisy_loss}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    '''
    def test_step(self, data):
        x, labels = data
        y = labels[:, :10]
        ground_truth = tf.cast(tf.math.argmax(labels[:, 10:], axis=1), tf.float32)
        labels_idx = tf.math.argmax(y, axis=1)
        labels_idx = tf.cast(labels_idx, tf.float32)

        noisy_y = tf.gather_nd(y, tf.where(ground_truth != labels_idx))
        noisy_x = tf.gather_nd(x, tf.where(ground_truth != labels_idx))
        clean_y = tf.gather_nd(y, tf.where(ground_truth == labels_idx))
        clean_x = tf.gather_nd(x, tf.where(ground_truth == labels_idx))

        logits_noisy_y = self(noisy_x, training=False)
        dist_noisy_y = tf.nn.softmax(logits_noisy_y)
        logits_clean_y = self(clean_x, training=False)
        dist_clean_y = tf.nn.softmax(logits_clean_y)

        self.cat_accuracy.update_state(dist_noisy_y, noisy_y)
        acc_noisy = self.cat_accuracy.result()
        self.cat_accuracy.reset_state()

        self.cat_accuracy.update_state(dist_clean_y, clean_y)
        acc_clean = self.cat_accuracy.result()
        self.cat_accuracy.reset_state()

        logits_y = self(x, training=False)
        dist_y = tf.nn.softmax(logits_y)
        self.cat_accuracy.update_state(dist_y, y)
        acc = self.cat_accuracy.result()
        self.cat_accuracy.reset_state()

        return {"acc": acc, "noisy_accuracy": acc_noisy, "clean_accuracy": acc_clean}
    '''
