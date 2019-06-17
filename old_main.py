# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018
@author: yjwu
Python 3.5.2
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
from spiking_model import*
from graphviz import Digraph

from args import get_parser

parser = get_parser()
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'spiking_model'
data_path =  './data/raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.base_batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.base_batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = OldSCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=args.learning_rate)


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


for epoch in range(args.num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images)
        # print("outputs is: ", outputs)
        # g = make_dot(outputs)
        # g.render('old_train.gv')
        labels_ = torch.zeros(args.base_batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        # print("the loss type is: ", type(loss))
        loss.backward()
        print("snn.fc2.weight.grad: ", snn.fc2.weight.grad.max())
        optimizer.step()
        if (i+1)%100 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, args.num_epochs, i+1, len(train_dataset)//args.base_batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, args.learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(args.base_batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    # if epoch % 5 == 0:
    #     print(acc)
    #     print('Saving..')
    #     state = {
    #         'net': snn.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'acc_record': acc_record,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt' + names + '.t7')
    #     best_acc = acc