#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

def add_conv_layer(inputs, filters, kernel_size, stride, padding):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, activation="relu")(inputs)


def add_conv_batchnorm_layer(inputs, filters, kernel_size, stride, padding):
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, activation=None, use_bias=False)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    return tf.keras.layers.Activation("relu")(out)


def add_max_pool_layer(inputs, kernel_size, stride):
    return tf.keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)(inputs)


def add_flatten_layer(inputs):
    return tf.keras.layers.Flatten()(inputs)


def add_dense_layer(inputs, hidden_layer_size):
    # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
    return tf.keras.layers.Dense(hidden_layer_size, activation="relu")(inputs)


def add_layer(param_string, inputs):
    l_type, params = param_string.split("-")[0], param_string.split("-")[1:]
    print("Adding layer {} with params {}".format(l_type, str(params)))

    if l_type == "C":
        filters, kernel_size, stride, padding = params
        filters, kernel_size, stride = int(filters), int(kernel_size), int(stride)
        return add_conv_layer(inputs, filters, kernel_size, stride, padding)
    elif l_type == "CB":
        filters, kernel_size, stride, padding = params
        filters, kernel_size, stride = int(filters), int(kernel_size), int(stride)
        return add_conv_batchnorm_layer(inputs, filters, kernel_size, stride, padding)
    elif l_type == "M":
        kernel_size, stride = map(int, params)
        return add_max_pool_layer(inputs, kernel_size, stride)
    elif l_type == "F":
        return add_flatten_layer(inputs)
    elif l_type == "D":
        hidden_layer_size = int(params[0])
        return add_dense_layer(inputs, hidden_layer_size)

    elif l_type == "R":
        print(param_string)
        layers_spec = param_string[param_string.index("[")+1:param_string.index("]")].split(",")
        hidden = inputs
        for layer_param in layers_spec:
            hidden = add_layer(layer_param, hidden)

        return tf.keras.layers.Add()([hidden, inputs])

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = inputs

        for layer_param in re.findall(r"C-[^,]*,*|R-\[[^]]*\],*|F,*|D-[^,]*,*|CB-[^,]*,*|M-[^,]*,*", args.cnn):
            if layer_param[-1] == ",":
                layer_param = layer_param[:-1]
            hidden = add_layer(layer_param, hidden)


        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    # parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
    parser.add_argument("--cnn", default="C-8-3-5-valid,R-[C-8-3-1-same,C-8-3-1-same],F,D-50", type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
