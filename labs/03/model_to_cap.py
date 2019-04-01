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


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="models\\97.517__uppercase-solution2.py-alph_size=60,bs=2776,ep=30,hid_ls=2,hid_l_size=76,window=20,lr_s=0.14,lr_dec=0.005.h5", type=str, help="Model file")
parser.add_argument("--window", default=20, type=int, help="Size of window")
parser.add_argument("--alphabet_size", default=60, type=int, help="Size of the alphabet")
parser.add_argument("--hidden_layers", default=2, type=int, help="Size of the alphabet")
parser.add_argument("--hidden_layer_size", default=76, type=int, help="Size of the alphabet")
parser.add_argument("--dropout", default=0.5, type=int, help="Size of the alphabet")

args = parser.parse_args()


# Load data
print("Loading UppercaseData...")
uppercase_data = UppercaseData(args.window, args.alphabet_size)
print("Data loaded!")



model = tf.keras.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(rate=args.dropout))
for i in range(args.hidden_layers):
    model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=args.dropout))

model.add(tf.keras.layers.Dense(uppercase_data.LABELS, activation=tf.nn.softmax))

model.load_weights(args.model_path)



with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    predictions = model.predict(uppercase_data.test.data["windows"])
    for i in range(len(predictions)):
        p = list(predictions[i])
        cap = p.index(max(p))
        letter = uppercase_data.test.text[i]
        if cap:
            out_file.write(letter.upper())
        else:
            out_file.write(letter)

    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.

