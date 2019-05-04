#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset
import math
import time
# 7f0a197b-bc00-11e7-a937-00505601122b
# 7cf40fc2-b294-11e7-a937-00505601122b



class Network:
    def __init__(self, pdt, args, num_words, num_chars, num_tags, num_anal_tags):

        self.learning_rate = 0.001
        # TODO: Define a suitable model.
        # TODO(we): Implement a one-layer RNN network. The input
        # `word_ids` consists of a batch of sentences, each
        # a sequence of word indices. Padded words have index 0.
        word_ids = tf.keras.Input(shape=(None,), dtype='int32', name="word_ids")



        charseqs = tf.keras.Input(shape=(None,), name="charseqs")
        charseq_ids = tf.keras.Input(shape=(None,), dtype='int32', name="charseq_ids")


        pos_ids = tf.keras.Input(shape=(None, None), dtype='int32', name="pos_ids")
        # pos_ids (Batch_size, max_sentence_len, max_word_pos_len)





        # TODO: Apart from `word_ids`, RNN CLEs utilize two more
        # inputs, `charseqs` containing unique words in batches (each word
        # being a sequence of character indices, padding characters again
        # have index 0) and `charseq_ids` with the same shape as `word_ids`,
        # but with indices pointing into `charseqs`.

        # TODO: Embed the characters in `charseqs` using embeddings of size
        # `args.cle_dim`, masking zero indices. Then, pass the embedded characters
        # through a bidirectional GRU with dimension `args.cle_dim`, concatenating
        # results in different dimensions.

        chars_embedding = tf.keras.layers.Embedding(
            num_chars, args.cle_dim)(charseqs)
        gru_cell = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(args.cle_dim,
                                kernel_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg),
                                recurrent_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg),
                                bias_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg)
                                ), merge_mode="sum")(chars_embedding)


        anal_tags_embedding = tf.keras.layers.Embedding(num_anal_tags, args.anal_pos_embedding_dim)(pos_ids)

        # pos_ids               (batch_size, words_in_sentence, n_postags)
        # anal_tags_embedding   (batch_size, words_in_sentence, n_postags, postag_dim)

        anal_tags_gru = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.anal_pos_embedding_dim), merge_mode="sum"))(anal_tags_embedding)
        # anal_tags_gru         (batch_size, words_in_sentence, anal_pos_embedding_dim)




        # Then, copy the computed embeddings of unique words to the correct sentence
        # positions. To that end, use `tf.gather` operation, which is given a matrix
        # and a tensor of indices, and replace each index by a corresponding row
        # of the matrix. You need to wrap the `tf.gather` in `tf.keras.layers.Lambda`
        # because of a bug [fixed 6 days ago in the master], so the call shoud look like
        # `tf.keras.layers.Lambda(lambda args: tf.gather(*args))(...)`
        # TODO(we): Embed input words with dimensionality `args.we_dim`, using
        # `mask_zero=True`.
        gather = tf.keras.layers.Lambda(lambda args: tf.gather(*args))([gru_cell, charseq_ids])

        # TODO: Concatenate the WE and CLE embeddings (in this order).
        self.words_embedding_layer = tf.keras.layers.Embedding(num_words, args.we_dim, trainable=args.trainable_word_embedding)
        # self.words_embedding_layer.trainable = args.trainable_word_embedding
        words_embedding = self.words_embedding_layer(word_ids)
        con = tf.keras.layers.Concatenate()([words_embedding, gather, anal_tags_gru])
        con = tf.keras.layers.Dropout(args.dropout)(con)

        # TODO(we): create specified `args.rnn_cell` rnn cell (lstm, gru) with
        # dimension `args.rnn_cell_dim` and apply it in a bidirectional way on
        # the embedded words, concatenating opposite directions.
        if args.rnn_cell == "LSTM":
            cell = tf.compat.v1.keras.layers.CuDNNLSTM
        elif args.rnn_cell == "GRU":
            cell = tf.keras.layers.GRU

        x = con

        recur_cell = cell(args.rnn_cell_dim,
                          kernel_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg),
                          recurrent_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg),
                          bias_regularizer=tf.keras.regularizers.L1L2(l2=args.l2_reg),
                          return_sequences=True)
        x = tf.keras.layers.Bidirectional(recur_cell, merge_mode="sum")(x)

        for i in range(args.reccurent_layers - 1 ):
            recur_cell = cell(args.rnn_cell_dim, return_sequences=True)
            x1 = tf.keras.layers.Bidirectional(recur_cell, merge_mode="sum")(x)
            x = tf.keras.layers.Add()([x, x1])

        if args.dropout_before_softmax:
            x = tf.keras.layers.Dropout(args.dropout_before_softmax)(x)

        # TODO(we): Add a softmax classification layer into `num_tags` classes, storing
        # the outputs in `predictions`.
        predictions = tf.keras.layers.Dense(
            num_tags, activation=tf.keras.activations.softmax)(x)

        self.model = tf.keras.Model(
            inputs=[word_ids, charseq_ids, charseqs, pos_ids], outputs=predictions)

        print("loading word embedding weights...")
        self.load_word_embedding_weights(pdt)

        tf.keras.utils.plot_model(self.model, 'tagger_cle_rnn_model.png', show_shapes=True)

        # TODO: Create an Adam optimizer in self._optimizer
        self._optimizer = tf.keras.optimizers.Adam()
        # TODO: Create a suitable loss in self._loss
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # TODO: Create two metrics in self._metrics dictionary:
        #  - "loss", which is tf.metrics.Mean()
        #  - "accuracy", which is suitable accuracy
        self._metrics = {
            "loss": tf.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy()}
        self._writer = tf.summary.create_file_writer(
            args.logdir, flush_millis=10 * 1000)

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

        self.ensemble_predictions = {"dev":[], "test":[]}



    def load_word_embedding_weights(self, pdt):
        if args.pretrained_file:
            print("Loading pretrained word embeddings into Embedding layer")
            weights_var = self.words_embedding_layer.variables[0]
            weights_var.assign(pdt.embedding_matrix)
        else:
            print("No pretrained embedding file specified!")

    def lr_schedule(self, epoch):
        if epoch < 25:
            return 0.001
        elif epoch < 40:
            return 0.0004
        elif epoch < 50:
            return 0.0001
        else:
            return 0.00005


    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3 + [tf.TensorSpec(shape=[None, None, None], dtype=tf.int32)],
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        # TODO: Generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        mask = tf.not_equal(tags, tf.constant(0, dtype=tags.dtype))

        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            # TODO: Compute `loss` using `self._loss`, passing the generated
            # tag mask as third parameter.
            loss = self._loss(tags, probabilities, mask)
            gradients = tape.gradient(loss, self.model.variables)
            self._optimizer.apply_gradients(
                zip(gradients, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                    # passing the tag mask as third argument.
                    metric(tags, probabilities, mask)
                tf.summary.scalar("train/{}".format(name),
                                  metric.result(), step=None)

    def train_epoch(self, dataset, args, epoch_i):
        # load and set learning rate
        try:
            self.learning_rate = self.read_learning_rate()
            print("Loaded learning_rate from file")
        except:
            self.learning_rate = self.lr_schedule(epoch_i)
            print("Error reading learning rate from file")

        print("Learning rate:", self.learning_rate)
        self._optimizer.lr = self.learning_rate

        batch_i = 0
        for batch in dataset.batches(args.batch_size):
            batch_i += 1
            print("\rBatch {}/{}".format(batch_i, math.ceil(dataset.size()/args.batch_size)), end="")
            self.train_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids, batch[dataset.FORMS].charseqs, batch[dataset.FORMS].anal_tags],
                             batch[dataset.TAGS].word_ids)

    def train(self, pdt, args):
        for i in range(args.epochs):
            print("\nEpoch {}".format(i+1))
            self.train_epoch(pdt.train, args, i+1)

            # train metrics
            metrics = self.evaluate(pdt.train, "train", args)
            print("\nTrain: " + ", ".join([key + ": " + str(metrics[key]) for key in metrics]))

            # dev metrics
            metrics = self.evaluate(pdt.dev, "dev", args)
            print("Dev:" + ", ".join([key + ": " + str(metrics[key]) for key in metrics]))

            # TODO evaluate ensemble and shift it
            # self.shift_ensemble(self.ensemble_predictions["dev"], args)
            # self.shift_ensemble(self.ensemble_predictions["train"], args)
            # self.evaluate_ensemble()


            if (i+1) % args.export_every == 0:
                self.export_test_annotation(pdt)


    def evaluate_batch(self, inputs, tags):
        # TODO: Again generate a mask from `tags` containing ones in positions
        # where tags are nonzero (using `tf.not_equal`).
        mask = tf.not_equal(tags, tf.constant(0, dtype=tags.dtype))
        probabilities = self.model(inputs, training=False)
        # TODO: Compute `loss` using `self._loss`, passing the generated
        # tag mask as third parameter.
        loss = self._loss(tags, probabilities, mask)

        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                # TODO: Update the `metric` using gold `tags` and generated `probabilities`,
                # passing the tag mask as third argument.
                metric(tags, probabilities, mask)

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            # TODO: Evaluate the given match, using the same inputs as in training.
            self.evaluate_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                                 batch[dataset.FORMS].charseqs, batch[dataset.FORMS].anal_tags], batch[dataset.TAGS].word_ids)

        metrics = {name: float(metric.result())
                   for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(
                    "{}/{}".format(dataset_name, name), value, step=None)

        return metrics


    def evaluate_ensemble(self, ensemble_pred, dataset, args):
        for sentence_index, sentence in enumerate(dataset[dataset.TAGS].word_ids):
            for word_index, tag in enumerate(sentence):
                ensemble_prob = np.zeros(len(dataset[dataset.TAGS].words))
                for ensemble in ensemble_pred:
                    pass
                    # TODO
        prob = sum(self.ensemble_predictions)




    def shift_ensemble(self, ensemble_pred, dataset, args):
        if len(self.ensemble_predictions) == args.ensemble_size:
            ensemble_pred = ensemble_pred[1:]
            ensemble_pred.append(self.predict_prob(dataset, args))


    # @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3])
    def predict_prob(self, dataset, args):
        batch_i = 0
        prediction_prob = []
        for batch in dataset.batches(args.batch_size):
            batch_i += 1
            print("\rPredicting batch prob {}/{}".format(batch_i, math.ceil(dataset.size() / args.batch_size)), end="")
            prediction_prob += self.predict_batch_prob([batch[0].word_ids, batch[0].charseq_ids, batch[0].charseqs, batch[0].anal_tags])

        return prediction_prob

    def predict_batch_prob(self, inputs):
        return self.model(inputs, training=False)

    def predict(self, dataset, args):
        # TODO: Predict method should return a list, each element corresponding
        # to one sentence. Each sentence should be a list/np.ndarray
        # containing _indices_ of chosen tags (not the logits/probabilities).
        prediction = []

        batch_i = 0
        for batch in dataset.batches(args.batch_size):
            batch_i += 1
            print("\rPredicting batch {}/{}".format(batch_i, math.ceil(dataset.size()/args.batch_size)), end="")

            prediction += self.predict_batch([batch[0].word_ids, batch[0].charseq_ids, batch[0].charseqs, batch[0].anal_tags])
            # print(prediction)
        print()

        return prediction

    def predict_batch(self, inputs):
        prediction  = []

        prob = self.model(inputs, training=False)

        for pred in prob:
            sentence = []
            for word in pred:
                sentence.append(np.argmax(word))
            prediction.append(sentence)
        return prediction

    def read_learning_rate(self):
        with open("learning_rate.txt", "r") as f:
            lr = float(f.read())
        return lr

    def export_test_annotation(self, pdt):
        # Generate test set annotations, but in args.logdir to allow parallel execution.
        metrics = self.evaluate(pdt.dev, "dev", args)
        accuracy = metrics["accuracy"]
        filename = str(accuracy) + "_" + model_name + ".txt"
        print("Exporting to:", filename)
        out_path = os.path.join("exported", filename)
        # if os.path.isdir(args.logdir): out_path = os.path.join(args.logdir, out_path)
        with open(out_path, "w", encoding="utf-8") as out_file:
            for i, sentence in enumerate(network.predict(morpho.test, args)):
                # print(sentence)
                # print("Len:", len(sentence), sentence)
                # print("Len word:", len(morpho.test.data[morpho.test.FORMS].word_strings[i]),
                #       morpho.test.data[morpho.test.FORMS].word_strings[i])
                for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                    print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                          morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                          morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                          sep="\t", file=out_file)
                print(file=out_file)


# FORMS - původní slova
# LEMMAS - slova v základním tvaru
# TAGS - slovní druhy

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--export_every", default=10, type=int, help="Export annotations every...")


    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--cle_layers", default=1,  type=int, help="Number of cle rnn layers")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=32, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=300, type=int, help="Word embedding dimension.")
    parser.add_argument("--reccurent_layers", default=4, type=int, help="Number of reccurent layers")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
    parser.add_argument("--dropout_before_softmax", default=0.5, type=float)
    parser.add_argument("--pretrained_file", default="c:/Users/Jakub/Downloads/only_used_small.vec", type=str, help="File with pretrained word embeddings")
    parser.add_argument("--trainable_word_embedding", default="yes", type=str, help="Whether or not word embedding layer is trainable")
    parser.add_argument("--l2_reg", default=0.01, type=float)
    parser.add_argument("--anal_pos_embedding_dim", default=16, type=int)
    parser.add_argument("--label_smoothing", default=0.1, type=float)


    """
    parser.add_argument("--cle_dim", default=75, type=int, help="CLE embedding dimension.")
    parser.add_argument("--max_sentences", default=0, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=400, type=int, help="RNN cell dimension.")
    parser.add_argument("--we_dim", default=300, type=int, help="Word embedding dimension.")
    parser.add_argument("--reccurent_layers", default=3, type=int, help="Number of reccurent layers")
    parser.add_argument("--dropout", default=0.3, type=float, help="Number of reccurent layers")
    parser.add_argument("--pretrained_file", default=None, type=str, help="File with pretrained word embeddings")
    parser.add_argument("--ensemble_size", default=5, type=int, help="File with pretrained word embeddings")
    """


    args = parser.parse_args()
    if args.trainable_word_embedding != "yes":
        print("Freeze word embedding")
        args.trainable_word_embedding = False
    else:
        args.trainable_word_embedding = True

    print("args.trainable_word_embedding", args.trainable_word_embedding)

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    model_name = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()))).replace("/", "_").replace(":", "_")

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        model_name))

    # Load the data. Using analyses is only optional.
    print("Loading czech_pdt dataset...")
    if args.max_sentences:
        morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences, pretrained=args.pretrained_file)
    else:
        morpho = MorphoDataset("czech_pdt", pretrained=args.pretrained_file)

    m = morpho.train.data[0]
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    batch = list(morpho.train.batches(2))[0]

    # Create the network and train
    print("Building netowrk...")
    network = Network(morpho, args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet),
                      num_anal_tags=len(morpho.pos_map))

    print("Starting training")
    try:
        network.train(morpho, args)
    except KeyboardInterrupt:
        network.export_test_annotation(morpho)


