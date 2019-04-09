# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib

from tensorflow.keras.models import load_model
from resnet import buildResnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cifar10 import CIFAR10


def string_to_tuple(string):
    return tuple(map(int, string.replace("(", "").replace(")", "").split(",")))

def parse_params(filename):
    global label_smoothing, layer1_params, res_layer_params, reg

    params = list(map(lambda x: x.split("=")[1], filename.replace(".h5", "").split("-")[1].split(";")))
    label_smoothing = float(params[0])
    layer1_params = string_to_tuple(params[1])
    res_layer_params = string_to_tuple(params[2])
    reg = float(params[3])


# Parse arguments
parser = argparse.ArgumentParser()

# parser.add_argument("--model_file", default="90.300_resnet_train.py-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,48,37);reg=0.00025.h5", type=str, help="Model path")
# parser.add_argument("--model_file", default="90.300_resnet_train.py-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,32,35);reg=0.00025.h5", type=str, help="Model path")
parser.add_argument("--model_file", default="resnetbig-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,32,25);reg=0.0001.h5", type=str, help="Model path")

args = parser.parse_args()
parse_params(args.model_file)

# define filepath parms
check_point_file = r"./densenet_check_point.h5"
loss_trend_graph_path = r"./loss.jpg"
acc_trend_graph_path = r"./acc.jpg"


cifar = CIFAR10()

model = buildResnet(input_shape=(32, 32, 3), layer1_params=layer1_params, res_layer_params=res_layer_params, reg=reg)
model.load_weights(args.model_file)


print((sum(np.array([[np.argmax(probs)] for probs in model.predict(cifar.dev.data["images"], batch_size=128)]) == cifar.dev.data["labels"])) / len(cifar.dev.data["labels"]))

np.save(os.path.join("exported_test", args.model_file), model.predict(cifar.test.data["images"], batch_size=128))


# with open(os.path.join("results", args.model_file + ".txt"), "w", encoding="utf-8") as out_file:
#     for probs in model.predict(cifar.test.data["images"], batch_size=128):
#         print(np.argmax(probs), file=out_file)

