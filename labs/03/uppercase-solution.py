#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

def convert_to_text(window):
    return "".join(list(map(lambda s: uppercase_data.train.alphabet[s], window)))


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=32, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=4096, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500, 500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=20, type=int, help="Window size to use.")
parser.add_argument("--smaller_data", default=1, type=float, help="Make train data smaller for testing purposes")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")

args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
model_name = "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()))))
args.logdir = os.path.join("logs", model_name)

# Load data
print("Loading UppercaseData...")
uppercase_data = UppercaseData(args.window, args.alphabet_size)
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

model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(rate=args.dropout))
for hidden in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=args.dropout))

model.add(tf.keras.layers.Dense(uppercase_data.LABELS, activation=tf.nn.softmax))


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None


try:
    model.fit(uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=[tb_callback],
              validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
              )
except KeyboardInterrupt:
    print("model saved!")
    model.save(os.path.join("models", model_name + ".h5"))

#
# test_logs = model.evaluate(
#     uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"] )
# print("Test logs: ", test_logs)

model.save(os.path.join("models", model_name + ".h5"))


with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    pass
