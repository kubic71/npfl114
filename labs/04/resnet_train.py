# -*- coding:utf-8 -*-
# 7f0a197b-bc00-11e7-a937-00505601122b
# 7cf40fc2-b294-11e7-a937-00505601122b
import tensorflow as tf
import numpy as np
import os
import matplotlib
import keras
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model

from tensorflow.keras.models import load_model
from resnet2 import buildResnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cifar10 import CIFAR10



batch_size = 64
nb_epoch = 10000
label_smoothing=0.1
input_shape=(32, 32, 3)


# (conv_size, nb_filters, strides)
layer1_params=(3,128,2)

# (conv size, num res layer filters, num res stages)
res_layer_params=(3,32,25)
reg=0.00075

use_augmentation = True
short_learning = True
decrease_lr_on_plateau = True
resume = False


model_name = "{}-{}".format(
    os.path.basename(__file__),
    "label_smoothing={};layer1_params={};res_layer_params={};reg={}".format(label_smoothing, layer1_params, res_layer_params, reg).replace(" ", ""))
logdir = os.path.join("logs", model_name)

# define filepath parms
check_point_file = r"./densenet_check_point.h5"
loss_trend_graph_path = r"./loss.jpg"
acc_trend_graph_path = r"./acc.jpg"


cifar = CIFAR10()

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


def lr_scheduler(epoch):
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

    # return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

plateau_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=np.sqrt(0.1),
                                cooldown=0, patience=5, min_lr=1e-5)


print('Now,we start compiling ResNet model...')

model = buildResnet(input_shape=input_shape, layer1_params=layer1_params, res_layer_params=res_layer_params, reg=reg)

if resume == True:
    model.load_weights(check_point_file)

optimizer = Adam()
# optimizer = SGD(lr=0.001)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
              optimizer=optimizer, metrics=['accuracy'])

print('Now,we start loading data...')

cifar.train.data["labels"] = tf.keras.utils.to_categorical(cifar.train.data["labels"])
cifar.dev.data["labels"] = tf.keras.utils.to_categorical(cifar.dev.data["labels"])

train_datagen = getDataGenerator()
train_datagen = train_datagen.flow(cifar.train.data["images"], cifar.train.data["labels"], batch_size=batch_size)

print('Now,we start defining callback functions...')


model_checkpoint = ModelCheckpoint(check_point_file, monitor="val_accuracy", save_best_only=True,
                                   save_weights_only=True, verbose=1)

tb_callback=tf.keras.callbacks.TensorBoard(os.path.join('logs', model_name), update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None


if decrease_lr_on_plateau:
    callbacks = [model_checkpoint, plateau_reducer, tb_callback]
else:
    callbacks = [model_checkpoint, reduce_lr, tb_callback]

print("Now,we start training...")
try:
    if use_augmentation:
        history = model.fit_generator(generator=train_datagen,
                                  steps_per_epoch=cifar.train.data["images"].shape[0] // batch_size,
                                  epochs=nb_epoch,
                                  callbacks=callbacks,
                                  validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
                                  # validation_steps=x_test.shape[0] // batch_size,
                                  verbose=1
                                  )
    else:
        history = model.fit(cifar.train.data["images"], cifar.train.data["labels"], epochs=nb_epoch, callbacks=callbacks,
                                  validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
                                  # validation_steps=x_test.shape[0] // batch_size,
                                  verbose=1, batch_size=batch_size)
except:
    pass

test_logs = model.evaluate(cifar.dev.data["images"], cifar.dev.data["labels"])
print(test_logs)
accuracy = test_logs[1]


model.save("{:.3f}".format(100 * accuracy)+ "_" + model_name + ".h5")


print("Now,we start drawing the loss and acc trends graph...")
# summarize history for accuracy
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)

# summarize history for loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)

print("We are done, everything seems OK...")


# if __name__ == '__main__':
#     set_max_gpu_memory()
# main(resume=True)
