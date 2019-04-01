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

def get_model_predictions(model_name, model_path):
    model_name = model_name.replace(".h5", "").split("-")[3]
    m = model_name.split(",")
    alphabet_size = int(m[0].split("=")[1])
    hidden_layers = int(m[3].split("=")[1])
    hidden_layer_size = int(m[4].split("=")[1])
    window = int(m[5].split("=")[1])

    print("Loading UppercaseData...")
    uppercase_data = UppercaseData(window, alphabet_size)
    print("Data loaded!")


    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=[2 * window + 1], dtype=tf.int32))
    model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
    model.add(tf.keras.layers.Flatten())

    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(uppercase_data.LABELS, activation=tf.nn.softmax))
    model.load_weights(model_path)


    return model.predict(uppercase_data.dev.data["windows"])



# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--models_dir", default="ensemble", type=str, help="Models directory")

args = parser.parse_args()


# Load data

print("Loading UppercaseData...")
uppercase_data = UppercaseData(0, 0)
print("Data loaded!")


ensemble_pred = None
for model_name in os.listdir(args.models_dir):
    pred = get_model_predictions(model_name, os.path.join(args.models_dir, model_name))

    if ensemble_pred is None:
        ensemble_pred = pred
    else:
        ensemble_pred += pred


with open("uppercase_dev.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    for i in range(len(ensemble_pred)):
        p = list(ensemble_pred[i])
        cap = p.index(max(p))

        letter = uppercase_data.dev.text[i]
        if cap:
            out_file.write(letter.upper())
        else:
            out_file.write(letter)

    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.

