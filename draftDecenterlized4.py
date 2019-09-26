import torchvision
import torchvision.transforms as transforms
import torch

import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import nn_classes
import data_loader
import ps_functions
import SGD_custom

# select gpu
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(device)

# number of workers
N_w = 15
# number of neighbors
N_n = 4
# number of training samples
# Cifar10  50,000
# Fashin MNIST 60,000
N_s = 60000

batch = 64

trainloaders, testloader = data_loader.FMNIST_data(batch, N_w, N_s)

connectionMatrix = np.empty([15,N_n])
seperation = 2
for worker in range(N_w):
    index = worker - ((N_n / 2) * seperation)
    if index < 0:
        index += N_w
    for i in range(len(connectionMatrix[worker])):
        if index % N_w == worker:
            index += seperation
            connectionMatrix[worker][i] = int(index % N_w)
        else:
            connectionMatrix[worker][i] = int(index % N_w)
        index += seperation



w_index = 0
results = np.empty([1, 100])
res_ind = 0
nets = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
reserveNets = [nn_classes.MNIST_NET().to(device) for n in range(N_w)]
ps_model = nn_classes.MNIST_NET().to(device)
avg_model = nn_classes.MNIST_NET().to(device)


lr = 1e-1
momentum = 0
weight_decay = 1e-4
stdev = 0.05

criterions = [nn.CrossEntropyLoss() for n in range(N_w)]
optimizers = [SGD_custom.define_optimizer(nets[n], lr, momentum, weight_decay) for n in range(N_w)]
avg_Optimizer = SGD_custom.define_optimizer(avg_model,lr,momentum, weight_decay)


# initilize all weights equally

[ps_functions.synch_weight(nets[i], ps_model) for i in range(N_w)]
ps_functions.synch_weight(ps_model, avg_model)


runs = int(10000)
for r in tqdm(range(runs)):
    # index of the worker doing local SGD
    w_index = w_index % N_w
    for worker in range(N_w):
        for data in trainloaders[worker]:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizers[worker].zero_grad()
            preds = nets[worker](inputs)
            loss = criterions[worker](preds, labels)
            loss.backward()
            optimizers[worker].step()
            break

    for worker in range(N_w):
        ps_functions.synch_weight(reserveNets[worker], nets[worker])
    for worker in range(N_w):
        ps_functions.initialize_zero(nets[worker])
        totalRand = 0
        rand = abs(np.random.normal(1, stdev, N_n))
        normalizationFactor = sum(rand) / N_n
        for i in range(N_n):
          neighbor = int(connectionMatrix[worker][i])
          constant = (N_n +1) * normalizationFactor / rand[i]
          ps_functions.weight_accumulate(reserveNets[neighbor], nets[worker],constant) #PWdif & 6
        ps_functions.weight_accumulate(reserveNets[worker], nets[worker], N_n + 1)

    if r % 100 == 0:
        ps_functions.initialize_zero(ps_model)
        for n in range(N_w):
            ps_functions.weight_accumulate(nets[n], ps_model, N_w)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = ps_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        results[0][res_ind] = 100 * correct / total
        res_ind += 1
    if r == 5000:
        for n in range(N_w):
            ps_functions.lr_change(0.015,optimizers[n])
    if r == 7500:
        for n in range(N_w):
            ps_functions.lr_change(0.0025, optimizers[n])

    # moving to next worker
    w_index += 1
f = open('decenterilizedConvTestSTD-' + str(stdev) + '.txt', 'ab')
np.savetxt(f, (results), fmt='%.5f', encoding='latin1')
f.close()







