#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub
import keras

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from caltech42 import Caltech42


def lr_schedule(epoch, short_learning=False):

    if short_learning:
        if epoch < 2:
            return 0.01
        elif epoch < 6:
            return 0.001
        elif epoch < 8:
            return 0.0001
        else:
            return 0.00001

    # 1e.-3 / 1.e-4 / 1.e-5
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.001
    elif epoch < 30:
        return 0.0001
    elif epoch < 40:
        return 0.00001
    # if epoch < 50:
    #     return 0.001
    # elif epoch < 70:
    #     return 0.0001
    # else:
    #     return 0.00001

def getDataGenerator():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # rotation_range=15
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(cifar.train.data["images"])
    return datagen

# The neural network model
class Network:
    def __init__(self, args):
        # TODO: You should define `self.model`. You should use the following layer:
        # generuje feature vektory velikosti 1280
        # mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use

        inputs = tf.keras.layers.Input([224, 224, 3])
        mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        features = mobilenet(inputs, training=False)

        size = 2048
        # classification layer
        x = tf.keras.layers.Dense(size, activation="relu")(features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(size, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(caltech42.LABELS, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        #
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})
        # when serving only is False, the number of parameters is 3x larger when using Adam

        self.model.compile(
            tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        self.model.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.model.tb_callback.on_train_end = lambda *_: None

    def train(self, caltech42, args):
        for i in range(args.epochs):
            print("Epoch {}:".format(i+1))

            batch_i = 0
            for batch in caltech42.train.batches(size=args.batch_size):
                inputs = batch["images"]
                outputs = batch["labels"]

                print('\rBatch: ', batch_i, '/', int(caltech42.train.size / args.batch_size), sep='', end='')
                batch_logs = self.model.train_on_batch(inputs, outputs)
                print(batch_logs)
                batch_i += 1

                current_learning_rate = lr_schedule(i+1)
                tf.keras.backend.set_value(self.model.optimizer.lr, current_learning_rate)  # set new lr


            self.generate_test_prediction(caltech42, args)
            print("'\n Dev accuracy: ", self.validate(caltech42, args))


    def generate_test_prediction(self, caltech42, args):
        # Generate test set annotations, but in args.logdir to allow parallel execution.
        accuracy = self.validate(caltech42, args)
        with open(os.path.join("test", "{}_caltech42_competition_test.txt".format(accuracy)), "w", encoding="utf-8") as out_file:
            for batch in caltech42.test.batches(size=args.batch_size):
                pred = self.model.predict(batch["images"])
                for prob in pred:
                    print(np.argmax(prob), file=out_file)

    def validate(self, caltech42, args):
        correct = 0
        total = 0
        for batch in caltech42.dev.batches(size=args.batch_size):
            inputs = batch["images"]
            targets = batch["labels"]
            pred = self.model.predict(inputs)

            correct += np.sum(np.argmax(pred, axis=1) == targets)
            total += len(targets)

        return correct / total

    def predict(self, dataset, args):
        return self.model.predict(dataset["images"])




if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    im = Image.open("rottweiler.jpg")
    im = im.resize((224, 224))
    im = np.array(im)
    im = im / 255.0

    # Fix random seeds
    # np.random.seed(42)
    # tf.random.set_seed(42)
    # tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    # tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    caltech42 = Caltech42()

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)
    network.generate_test_prediction(caltech42, args)


