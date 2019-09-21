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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



w_index = 0
results = np.empty([1, 100])
res_ind = 0
nets = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
netsCurrent = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
netsOLD = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
netsDif = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
netsAvg = [nn_classes.MNIST_NET().to(device) for n in range(N_w)] #1
ps_model = nn_classes.MNIST_NET().to(device)
avg_model = nn_classes.MNIST_NET().to(device)


lr = 1e-1
momentum = 0
weight_decay = 1e-4

criterions = [nn.CrossEntropyLoss() for n in range(N_w)]
optimizers = [SGD_custom.define_optimizer(nets[n], lr, momentum, weight_decay) for n in range(N_w)]
avg_Optimizer = SGD_custom.define_optimizer(avg_model,lr,momentum, weight_decay)


# initilize all weights equally

[ps_functions.synch_weight(nets[i], ps_model) for i in range(N_w)]
[ps_functions.synch_weight(netsCurrent[i], ps_model) for i in range(N_w)]
[ps_functions.synch_weight(netsOLD[i], ps_model) for i in range(N_w)]
[ps_functions.synch_weight(netsDif[i], ps_model) for i in range(N_w)]
[ps_functions.synch_weight(netsAvg[i], ps_model) for i in range(N_w)]
ps_functions.synch_weight(ps_model, avg_model)


runs = int(10000)
for r in tqdm(range(runs)): # 2
    # index of the worker doing local SGD
    w_index = w_index % N_w
    for worker in range(N_w): #3
        for data in trainloaders[worker]:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizers[worker].zero_grad()
            preds = nets[worker](inputs)
            loss = criterions[worker](preds, labels)
            loss.backward()
            break
    #4
    for worker in range(N_w):
        ps_functions.weight_dif(netsCurrent[worker], netsOLD[worker], netsDif[worker])
    for worker in range(N_w):
        ps_functions.synch_weight(nets[worker],netsAvg[worker]) #5
        index = worker-N_n
        if index < 0:
         index += N_w
        for i in range(N_n+1):
            ps_functions.weight_accumulate(netsDif[int((index+(i*2)) % N_w)],nets[worker],N_n+1) #PWdif & 6
        ps_functions.synch_weight(netsAvg[worker],nets[worker]) #7
        optimizers[worker].step() #8
        ps_functions.synch_weight(netsOLD[worker],netsCurrent[worker]) #9
        ps_functions.synch_weight(netsCurrent[worker],nets[worker]) #10


    # w_index sends its model to other workers
    # # other workers upon receiving the model take the average

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
            ps_functions.lr_change(0.0015, optimizers[n])

    # moving to next worker
    w_index += 1
f = open('decenterilizedAlg2' + '.txt', 'ab')
np.savetxt(f, (results), fmt='%.5f', encoding='latin1')
f.close()







