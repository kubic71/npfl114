#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys

from mnist import MNIST

# The neural network model
class Network:
    def get_feature_extractor(self):
        net_input = tf.keras.Input(shape=(MNIST.W, MNIST.H, MNIST.C))
        x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, padding="valid", activation="relu")(net_input)
        x = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=2, padding="valid", activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        net_output = tf.keras.layers.Dense(200, activation="relu")(x)
        return tf.keras.Model(net_input, net_output, name="image_feature_extractor")

    def get_image_classifier(self):
        net_input = tf.keras.Input(shape=(200, ))
        net_output = tf.keras.layers.Dense(10, activation="softmax")(net_input)
        return tf.keras.Model(net_input, net_output, name="image_classifier")




    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.
        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation (the goal is to predict if the first digit is larger than the second)
        #
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.

        left_input = tf.keras.Input(shape=(MNIST.W, MNIST.H, MNIST.C), name="left_images")
        right_input = tf.keras.Input(shape=(MNIST.W, MNIST.H, MNIST.C), name="right_images")

        feature_extractor = self.get_feature_extractor()

        # apply the feature extractor with same weights to both of the input images
        left_features = feature_extractor(left_input)
        right_features = feature_extractor(right_input)

        image_clasifier = self.get_image_classifier()

        # name the classification output with tf.identity, so it can be referenced when specifying losses and training target data
        left_classification = image_clasifier(left_features)
        left_classification = tf.identity(left_classification, name="left_prediction")

        right_classification = image_clasifier(right_features)
        right_classification = tf.identity(right_classification, name="right_prediction")

        x = tf.keras.layers.Concatenate()([left_features, right_features])
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        comparator = tf.keras.layers.Dense(1, activation="sigmoid", name="comparator")(x)

        self.model = tf.keras.Model(inputs=[left_input, right_input], outputs=[left_classification, right_classification, comparator])

        # tf.keras.utils.plot_model(self.model, 'multi_input_and_output_model.png', show_shapes=True)
        # print("model image saved")

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss={'left_prediction': tf.keras.losses.SparseCategoricalCrossentropy(),
                            "right_prediction": tf.keras.losses.SparseCategoricalCrossentropy(),
                            "comparator": tf.keras.losses.BinaryCrossentropy()}
                           # metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                           )




    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                # TODO: yield the suitable modified inputs and targets using batches[0:2]
                model_inputs = {}
                model_outputs = {}

                model_inputs["left_images"] = batches[0]["images"]
                model_inputs["right_images"] = batches[1]["images"]

                model_outputs["left_prediction"] = batches[0]["labels"]
                model_outputs["right_prediction"] = batches[1]["labels"]
                model_outputs["comparator"] = np.array(list(map(int, model_outputs["left_prediction"] > model_outputs["right_prediction"])))

                yield (model_inputs, model_outputs)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):
                model_inputs, model_targes = batch
                self.model.train_on_batch(model_inputs, model_targes)

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))




    def evaluate(self, dataset, args):

        def to_sparse(arr):
            return np.array(list(map(np.argmax, arr)))


        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.

        total_indirect = 0
        total_direct = 0
        total = 0
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            left_prediction, right_prediction, compare = self.model.predict(inputs)

            indirect = to_sparse(left_prediction) > to_sparse(right_prediction)
            total_indirect += sum(indirect == targets["comparator"])
            total_direct += sum(np.round(compare.flatten()) == targets["comparator"])
            total += len(indirect)

        direct_accuracy = total_direct / total
        indirect_accuracy = total_indirect / total
        return direct_accuracy, indirect_accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=True, action="store_true", help="Evaluation in ReCodEx.")
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

    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
