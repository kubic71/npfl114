#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

# Parse arguments
# TODO: Set reasonable defaults and possibly add more arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--hidden_layer", default=4, type=int)
args = parser.parse_args()
lr =0.01
lrf=0.0001

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = "logs\\{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
)

# Load the data
observations, labels = [], []
with open("gym_cartpole-data.txt", "r") as data:
    for line in data:
        columns = line.rstrip("\n").split()
        observations.append([float(column) for column in columns[0:-1]])
        labels.append(int(columns[-1]))
observations, labels = np.array(observations), np.array(labels)

# TODO: Create the model in the `model` variable.
# However, beware that there is currently a bug in Keras which does
# not correctly serialize InputLayer. Instead of using an InputLayer,
# pass explicitly `input_shape` to the first real model layer.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu, input_shape=[4]),
    tf.keras.layers.Dense(2, activation=tf.nn.relu),
    # tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax),
])


# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
#             lr,
#             args.epochs,   # epochs * (size / batch_size)
#             lrf)),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr, args.epochs, lrf/lr)),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


tb_callback=tf.keras.callbacks.TensorBoard(args.logdir)
model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])

model.save("gym_cartpole_model.h5", include_optimizer=False)
