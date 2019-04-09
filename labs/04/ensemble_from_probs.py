import os
import numpy as np
import random

from cifar10 import CIFAR10
cifar = CIFAR10()
prob = None

dir = "exported_dev"

# weig = [0.3114083907696912, 0.619123705974144, 0.7559370141571807, 0.06138325037611192, 0.10029516426205498, 0.9480189822174048, 0.16771272937686788, 0.1187925046168814]
# weig = [0.6, 0.619123705974144, 0.7559370141571807, 0, 0.10029516426205498, 0.9480189822174048, 0.16771272937686788]
weig = [0.6514226996374671, 0.6704702600103003, 0.7085720260166704, 0.09789678949713831, 0.10975879636788388, 0.9816333798351524, 0.4017273770326337]

weights = [["90.300_resnet_train.py-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,32,35);reg=0.00025.h5.npy", 4],
           ["90.300_resnet_train.py-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,48,37);reg=0.00025.h5.npy", 0],
           ["resnetbig-label_smoothing=0.1;layer1_params=(3,128,2);res_layer_params=(3,32,25);reg=0.0001.h5.npy", 4],
           ["probs_dev_michal.npy", 0],
           ["probs_dev_vilda_a48.npy", 1],
           ["probs_dev_vilda_all_res.npy", 4],
           ["probs_dev_vilda_alt.npy", 1]]


for i, w in enumerate(weig):
    weights[i][1] = w

def random_acc():
    global weights
    r_weights = [random.random() for i in range(len(weights))]

    for i in range(len(weights)):
        weights[i][1] = r_weights[i]


    prob = None
    for model, weight in weights:
        if prob is not None:
            prob += np.load(os.path.join(dir, model)) * weight
        else:
            prob = np.load(os.path.join(dir, model)) * weight

    if dir == "exported_dev":
        return ((sum(np.array([[np.argmax(probs)] for probs in prob]) == cifar.dev.data["labels"])) / len(
            cifar.dev.data["labels"])), r_weights



def get_accuracy(i, j, k, l):
    global weights

    weights[3][1] = i
    weights[4][1] = j
    weights[5][1] = k
    weights[6][1] = l

    prob = None
    for model, weight in weights:
        if prob is not None:
            prob += np.load(os.path.join(dir, model)) * weight
        else:
            prob = np.load(os.path.join(dir, model)) * weight

    if dir == "exported_dev":
        return ((sum(np.array([[np.argmax(probs)] for probs in prob]) == cifar.dev.data["labels"])) / len(
            cifar.dev.data["labels"]))

m = 0
ijk = None
"""
for i in range(5):
    print(i)
    for j in range(5):
        for k in range(5):
            for l in range(5):

                # print(i, j, k)
                acc = get_accuracy(i, j, k, l)[0]
                if acc > m:
                    m = acc
                    ijk = (i,j,k, l)
                    print(m)


print("max ijk", ijk)

m, best = 0, None
while True:
    acc, r_w = random_acc()
    if acc > m:
        m = acc
        best = r_w
        print(m, r_w)

"""

for model, weight in weights:
    if prob is not None:
        prob += np.load(os.path.join(dir, model)) * weight
    else:
        prob = np.load(os.path.join(dir, model)) * weight


if dir == "exported_dev":
    print((sum(np.array([[np.argmax(probs)] for probs in prob]) == cifar.dev.data["labels"])) / len(cifar.dev.data["labels"]))

np.save("probs_dev", prob)