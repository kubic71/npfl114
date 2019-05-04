#!/usr/bin/env python3
import contextlib

import numpy as np
import tensorflow as tf
import keras

from timit_mfcc import TimitMFCC

class Network:
    def __init__(self, args, name):
        self._beam_width = args.ctc_beam
        self.name = name

        # TODO: Define a suitable model, given already masked `mfcc` with shape
        # `[None, TimitMFCC.MFCC_DIM]`. The last layer should be a Dense layer
        # without an activation and with `len(TimitMFCC.LETTERS) + 1` outputs,
        # where the `+ 1` is for the CTC blank symbol.
        #
        # Store the results in `self.model`.
        masked_mfcc = tf.keras.Input(shape=(None, TimitMFCC.MFCC_DIM))
        recur_cell = tf.compat.v1.keras.layers.CuDNNLSTM(args.rnn_cell_dim, return_sequences=True)
        bidir = tf.keras.layers.Bidirectional(recur_cell)(masked_mfcc)
        bidir = tf.keras.layers.Dropout(0.5)(bidir)

        recur_cell2 = tf.compat.v1.keras.layers.CuDNNLSTM(args.rnn_cell_dim, return_sequences=True)
        bidir = tf.keras.layers.Bidirectional(recur_cell2)(bidir)
        bidir = tf.keras.layers.Dropout(0.5)(bidir)

        recur_cell3 = tf.compat.v1.keras.layers.CuDNNLSTM(args.rnn_cell_dim, return_sequences=True)
        bidir = tf.keras.layers.Bidirectional(recur_cell3)(bidir)
        bidir = tf.keras.layers.Dropout(0.5)(bidir)

        self.num_outputs = len(TimitMFCC.LETTERS) + 1

        # x = tf.keras.layers.Dense(100, activation="relu")(bidir)
        # x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(self.num_outputs, activation=None)(bidir)

        # The following are just defaults, do not hesitate to modify them.
        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy() # ????
        self._metrics = {"loss": tf.metrics.Mean(), "edit_distance": tf.metrics.Mean()}
        self._batch_loss = None
        self._batch_edit_distance = None
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        self.model = tf.keras.Model(inputs=masked_mfcc, outputs=predictions)




        # ???
        # self.model.compile(optimizer=tf.optimizers.Adam(),
        #                    loss=tf.losses.SparseCategoricalCrossentropy(),
        #                    metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])


    # Converts given tensor with `0` values for padding elements, create
    # a SparseTensor.
    def _to_sparse(self, tensor):
        tensor_indices = tf.where(tf.not_equal(tensor, 0))
        return tf.sparse.SparseTensor(tensor_indices, tf.gather_nd(tensor, tensor_indices), tf.shape(tensor, tf.int64))

    # Convert given sparse tensor to a (dense_output, sequence_lengths).
    def _to_dense(self, tensor):
        tensor = tf.sparse.to_dense(tensor, default_value=-1)
        tensor_lens = tf.reduce_sum(tf.cast(tf.not_equal(tensor, -1), tf.int32), axis=1)
        return tensor, tensor_lens

    # Compute logits given input mfcc, mfcc_lens and training flags.
    # Also transpose the logits to `[time_steps, batch, dimension]` shape
    # which is required by the following CTC operations.
    def _compute_logits(self, mfcc, mfcc_lens, training):
        # logits = self.model(mfcc, mask=tf.sequence_mask(mfcc_lens), training=training)

        # For GPU
        logits = self.model(mfcc, training=training)

        return tf.transpose(logits, [1, 0, 2])

    # Compute CTC loss using given logits, their lengths, and sparse targets.
    def _ctc_loss(self, logits, logits_len, sparse_targets):
        loss = tf.nn.ctc_loss(sparse_targets, logits, None, logits_len, blank_index=len(TimitMFCC.LETTERS))
        self._metrics["loss"](loss)
        self._batch_loss = loss
        return tf.reduce_mean(loss)

    # Perform CTC predictions given logits and their lengths.
    def _ctc_predict(self, logits, logits_len):
        (predictions,), _ = tf.nn.ctc_beam_search_decoder(logits, logits_len, beam_width=self._beam_width)
        return tf.cast(predictions, tf.int32)

    # Compute edit distance given sparse predictions and sparse targets.
    def _edit_distance(self, sparse_predictions, sparse_targets):
        edit_distance = tf.edit_distance(sparse_predictions, sparse_targets, normalize=True)
        self._metrics["edit_distance"](edit_distance)
        self._batch_edit_distance = edit_distance
        return edit_distance

    def editDistanceStrings(self, s1, s2):
        # nevim jak dostat z _edit_distance cislo
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, TimitMFCC.MFCC_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, mfcc, mfcc_lens, targets):
        """
        The basic architecture:
        - converts target letters into sparse representation,
        - use a bidirectional RNN and an output linear layer without activation,
        - compute CTC loss (tf.nn.ctc_loss),
        - if required, perform decoding by a CTC decoder (tf.nn.ctc_beam_search_decoder) and possibly evaluate results using normalized edit distance (tf.edit_distance).
        """

        # convert target letters to sparse representation
        sparse_targets = self._to_sparse(targets)

        with tf.GradientTape() as tape:
            # mfcc is masked according to mfcc_lens
            logits = self._compute_logits(mfcc, mfcc_lens, training=True)

            loss = self._ctc_loss(logits, mfcc_lens, sparse_targets)
            gradients = tape.gradient(loss, self.model.variables)

            grad_norm = 0
            clipped_norm = 0

            before_grad_norm = print("clipping grad:, ", grad_norm)
            gradients, grad_norm = tf.clip_by_global_norm(gradients, args.clip_gradient)
            print("after:", grad_norm)


            self._optimizer.apply_gradients(
                zip(gradients, self.model.variables))





        sparse_predictions = self._ctc_predict(logits, mfcc_lens)
        edit_distance = self._edit_distance(sparse_predictions, sparse_targets)

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                elif name == "edit_distance":
                    metric(edit_distance)
                    # passing the tag mask as third argument.
                tf.summary.scalar("train/{}".format(name),
                                  metric.result(), step=None)

                tf.summary.scalar("train/gradient_norm", grad_norm)


    def train_epoch(self, dataset, args):
        # TODO: Store suitable metrics in TensorBoard
        # What metrics? loss and edit_distance?
        batch_n = 1
        for batch in dataset.batches(args.batch_size):
            self.train_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"])
            print("\rBatch {}/{}".format(batch_n, int(dataset.size / args.batch_size)), end="")
            batch_n += 1







    def evaluate_batch(self, mfcc, mfcc_lens, targets, letters_len):

        total_len = 0
        i = 0
        total_distance = 0
        for prediction, prediction_len in zip(*self.predict_batch(mfcc, mfcc_lens)):
            sentence = np.array(prediction[:prediction_len])
            gold = targets[i][:letters_len[i]]
            # print("sentence:", self.to_string(sentence))
            # print("gold: ", self.to_string(gold))
            total_distance += self.editDistanceStrings(sentence, gold)
            i += 1
            total_len += len(gold)


        return total_distance, total_len


    def evaluate(self, dataset, dataset_name, args):
        # TODO: Store suitable metrics in TensorBoard
        print("\nEvaluating...")
        total_len = 0
        total_distance = 0
        for batch in dataset.batches(args.batch_size):
            dist, length = self.evaluate_batch(batch["mfcc"], batch["mfcc_len"], batch["letters"], batch["letters_len"])
            total_len += length
            total_distance += dist

        print("total dist {}, total_len: {}, ratio: {}".format(total_distance, total_len, total_distance/total_len))
        self.generate_test_annotations(args, total_distance/total_len)
        return total_distance/total_len



    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, TimitMFCC.MFCC_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def predict_batch(self, mfcc, mfcc_lens):
        # TODO: Implement batch prediction, returning a tuple (dense_predictions, prediction_lens)
        # produced by self._to_dense.

        logits = self._compute_logits(mfcc, mfcc_lens, training=False)
        sparse_predictions = self._ctc_predict(logits, mfcc_lens)
        dense_prediction, prediction_len = self._to_dense(sparse_predictions)
        return(dense_prediction, prediction_len)


    def to_string(self, indeces):
        return "".join(list(map(lambda x: TimitMFCC.LETTERS[x], indeces)))

    def predict(self, dataset, args):
        sentences = []
        for batch in dataset.batches(args.batch_size):
            for prediction, prediction_len in zip(*self.predict_batch(batch["mfcc"], batch["mfcc_len"])):
                sentences.append(np.array(prediction[:prediction_len]))
        return sentences

    def generate_test_annotations(self, args, dev_dist):
        filename = str(dev_dist) + "_" + self.name

        out_path = os.path.join("test_results", filename)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for sentence in network.predict(timit.test, args):
                print(" ".join(timit.LETTERS[letters] for letters in sentence), file=out_file)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=40, type=int, help="Batch size.")
    parser.add_argument("--ctc_beam", default=16, type=int, help="CTC beam.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--rnn_cell_dim", default=256, type=int, help="RNN LSTM cell dimension")
    parser.add_argument("--clip_gradient", default=400, type=float, help="Norm for gradient clipping.")

    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    modelname = "{}-{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()))))
    args.logdir = os.path.join("logs", modelname)

    # Load the data
    timit = TimitMFCC()

    batch = list(timit.dev.batches(2))[0]


    # Create the network and train
    network = Network(args, modelname)
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch + 1))
        network.train_epoch(timit.train, args)
        network.evaluate(timit.dev, "dev", args)


    # Generate test set annotations, but to allow parallel execution, create it
    # in in args.logdir if it exists.