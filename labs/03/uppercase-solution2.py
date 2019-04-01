#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import random
import sys

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

def convert_to_text(window):
    return "".join(list(map(lambda s: uppercase_data.train.alphabet[s], window)))

def random_log(low, high):
    ll = np.log(low)
    lh = np.log(high)
    exp = random.uniform(ll, lh)
    return np.e ** exp


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size_low", default=10, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--alphabet_size_high", default=100, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size_low", default=16, type=int, help="Batch size.")
parser.add_argument("--batch_size_high", default=8192, type=int, help="Batch size.")
parser.add_argument("--epochs_low", default=5, type=int, help="Number of epochs.")
parser.add_argument("--epochs_high", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers_low", default=1, type=int, help="Hidden layer configuration.")
parser.add_argument("--hidden_layers_high", default=4, type=int, help="Hidden layer configuration.")
parser.add_argument("--hidden_layer_size_low", default=4, type=int, help="Hidden layer configuration.")
parser.add_argument("--hidden_layer_size_high", default=100, type=int, help="Hidden layer configuration.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window_low", default=5, type=int, help="Window size to use.")
parser.add_argument("--window_high", default=50, type=int, help="Window size to use.")
parser.add_argument("--smaller_data", default=1, type=float, help="Make train data smaller for testing purposes")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
parser.add_argument("--learning_rate_start_low", default=0.01, type=float, help="Dropout rate")
parser.add_argument("--learning_rate_start_high", default=1, type=float, help="Dropout rate")
parser.add_argument("--learning_rate_decay_low", default=0.001, type=float, help="Dropout rate")
parser.add_argument("--learning_rate_decay_high", default=0.5, type=float, help="Dropout rate")

args = parser.parse_args()
#args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# choose random params
# alphabet_size = int(random_log(args.alphabet_size_low, args.alphabet_size_high))
# batch_size = int(random_log(args.batch_size_low, args.batch_size_high))
# epochs = random.randint(args.epochs_low, args.epochs_high)
# hidden_layers = random.randint(args.hidden_layers_low, args.hidden_layers_high)
# hidden_layer_size = int(random_log(args.hidden_layer_size_low, args.hidden_layer_size_high))
# window = random.randint(args.window_low, args.window_high)
# learning_rate_start = random_log(args.learning_rate_start_low, args.learning_rate_start_high)
# learning_decay = random_log(args.learning_rate_decay_low, args.learning_rate_decay_high)

alphabet_size = 60
batch_size = 2776
epochs = 30
hidden_layers = 2
hidden_layer_size = 76
window = 20


learning_rate_start = 0.14
learning_decay = 0.005




# Create logdir name
model_name = "{}-{}".format(
    os.path.basename(__file__),
    "alph_size={},bs={},ep={},hid_ls={},hid_l_size={},window={},lr_s={},lr_dec={}".format(alphabet_size, batch_size, epochs, hidden_layers, hidden_layer_size, window, learning_rate_start, learning_decay))
args.logdir = os.path.join("logs", model_name)
print(model_name)

# Load data
print("Loading UppercaseData...")
uppercase_data = UppercaseData(window, alphabet_size)
print("Data loaded!")

size = uppercase_data.train.size
uppercase_data.train.data["windows"] = uppercase_data.train.data["windows"][:int(size*args.smaller_data)]
uppercase_data.train.data["labels"] = uppercase_data.train.data["labels"][:int(size*args.smaller_data)]

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.


model = tf.keras.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=[2 * window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
model.add(tf.keras.layers.Flatten())

for i in range(hidden_layers):
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=args.dropout))

model.add(tf.keras.layers.Dense(uppercase_data.LABELS, activation=tf.nn.softmax))



decay_steps = int(epochs * (uppercase_data.train.size / batch_size))   # or -1 ???

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate_start,
            decay_steps,
            decay_rate=learning_decay)


model.compile(
    # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    optimizer=tf.keras.optimizers.Adam(),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None


try:
    model.fit(uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
             callbacks=[tb_callback]
              )
except KeyboardInterrupt:
    pass
    # print("model saved!")
    # model.save(os.path.join("models", model_name + ".h5"))
    # sys.exit(0)

#
# test_logs = model.evaluate(
#     uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"] )
# print("Test logs: ", test_logs)


test_logs = model.evaluate(
    uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"])
print(test_logs)
accuracy = test_logs[1]
print("Model evaluated, accuracy: ", accuracy)

model.save(os.path.join("models", "{:.3f}".format(100 * accuracy)+ "__"  + model_name + ".h5"))
f = open(os.path.join("scores", "{:.3f}".format(100 * accuracy)+ "__" + model_name), "w")
f.close()

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    pass
