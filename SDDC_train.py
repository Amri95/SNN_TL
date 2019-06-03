# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 09:46:25 2018

@author: yjwu

Python 3.5.2

"""

from __future__ import print_function
import math
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import dataloader
import os
import time
from spiking_model import*
from graphviz import Digraph


# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'transfer_spiking_model'
ROOT_PATH = './data/Office31'
SOURCE_NAME = 'amazon'
TARGET_NAME = 'webcam'

device_ids = [0, 1, 2, 3]
batch_size = base_batch_size * len(device_ids)

# source_loader = dataloader.load_training(ROOT_PATH, SOURCE_NAME, batch_size)
# target_train_loader = dataloader.load_training(ROOT_PATH, TARGET_NAME, batch_size)
# target_test_loader = dataloader.load_testing(ROOT_PATH, TARGET_NAME, batch_size)

source_loader, target_test_loader = dataloader.load_data(ROOT_PATH, SOURCE_NAME, batch_size)
target_train_loader = dataloader.load_training(ROOT_PATH, TARGET_NAME, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = torch.nn.DataParallel(SDDC(), device_ids)
# snn = SDDC()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate, weight_decay=l2_decay)


def calculate_error_classification(spikes, des_spikes):
    err = (torch.sum(spikes, -1, keepdim=True) - des_spikes)
    return err


def calculate_l2_loss_classification(spikes, des_spikes):
    return torch.sum(calculate_error_classification(spikes, des_spikes) ** 2) / 2 * 1


def step_decay(epoch, learning_rate):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


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


def train_ddcnet(epoch, model, learning_rate, source_loader, target_loader):
    """
    train source and target domain on ddcnet
    :param epoch: current training epoch
    :param model: defined ddcnet
    :param learning_rate: initial learning rate
    :param source_loader: source loader
    :param target_loader: target train loader
    :return:
    """
    log_interval = 10
    # LEARNING_RATE = step_decay(epoch, learning_rate)
    # print('Learning Rate: ', LEARNING_RATE)
    # optimizer = optim.SGD([
    #     # {'params': model.features.parameters()},
    #     # {'params': model.classifier.parameters()},
    #     # {'params': model.bottleneck.parameters(), 'lr': LEARNING_RATE},
    #     # {'params': model.final_classifier.parameters(), 'lr': LEARNING_RATE},
    #     {'params': model.parameters(), 'lr': LEARNING_RATE}
    # ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    # enter training mode
    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)

    correct = 0
    total_loss = 0

    for i in range(1, num_iter):
        source_data, source_label = iter_source.next()

        target_data, _ = iter_target.next()
        if i % len(target_loader) == 0:
            iter_target = iter(target_loader)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        model.zero_grad()
        optimizer.zero_grad()

        source_preds, mmd_loss = model(source_data, target_data)
        # print(mmd_loss)
        # g = make_dot(mmd_loss)
        # g.render('tl-graph_1.gv')
        # break
        # print(source_label)
        source_label_ = torch.zeros(batch_size, num_classes).cuda().scatter_(1, source_label.view(-1, 1), 1)
        # print(source_label_)
        _, predicted = source_preds.max(1)
        correct += float(predicted.eq(source_label).sum().item())

        # preds = source_preds.data.max(1, keepdim=True)[1]
        # correct += preds.eq(source_label.data.view_as(preds)).sum()

        # print(source_preds.shape, source_label_.shape)

        # clf_loss = calculate_l2_loss_classification(source_preds, source_label_)
        clf_loss = criterion(source_preds, source_label_)  # clf_criterion(source_preds, source_label)
        # print(clf_loss)
        # clf_loss.backward()
        loss = clf_loss  # + 0.25 * mmd_loss
        total_loss += clf_loss.item()

        # print("source_data.requires_grad:")
        # print(source_data.requires_grad)  # False
        # print("the grad of source_data:")
        # print(source_data.grad)  # None
        # print("targete_data.requires_grad:")
        # print(target_data.requires_grad)  # False
        # print("the grad of target_data:")
        # print(target_data.grad)  # None
        loss.backward()  # retain_graph=True)
        # print("the grad of net:")
        # print(model.conv0.bias.grad)  # no-None
        # print("source_data.requires_grad:")
        # print(source_data.requires_grad)  # False
        # print("the grad of source_data:")
        # print(source_data.grad)  # None
        # print("targete_data.requires_grad:")
        # print(target_data.requires_grad)  # False
        # print("the grad of target_data:")
        # print(target_data.grad)  # None

        optimizer.step()

        if i % log_interval == 4:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_loader) * batch_size,
                100. * i / len(source_loader), loss, clf_loss.item(), mmd_loss.item()))

    total_loss /= len(source_loader)
    acc_train = float(correct) * 100. / (len(source_loader) * batch_size)

    print('{} set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        SOURCE_NAME, total_loss, correct, len(source_loader.dataset), acc_train))


def test_ddcnet(model, target_loader):
    """
    test target data on fine-tuned alexnet
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """

    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        target_preds, _ = model(data, data)

        _, predicted = target_preds.max(1)
        correct += float(predicted.eq(target).sum().item())

        target_ = torch.zeros(batch_size, num_classes).cuda().scatter_(1, target.view(-1, 1), 1)
        test_loss += criterion(target_preds, target_)  # sum up batch loss

        # test_loss += clf_criterion(target_preds, target)  # sum up batch loss
        # pred = target_preds.data.max(1)[1]  # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(target_loader)
    print('{} set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        TARGET_NAME, test_loss.item(), correct, len(target_loader.dataset),
        # TARGET_NAME, test_loss.data[0], correct, len(target_loader.dataset),
        100. * correct / len(target_loader.dataset)))

    return correct


if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):
        print('Train Epoch: ', epoch)
        train_ddcnet(epoch, snn, learning_rate, source_loader, target_train_loader)
        with torch.no_grad():
            correct = test_ddcnet(snn, target_test_loader)
            # acc_record.append(correct)
            # if epoch % 5 == 0:
            #     print(correct)
            #     print('Saving..')
            #     state = {
            #         'net': snn.state_dict(),
            #         'acc': correct,
            #         'epoch': epoch,
            #         'acc_record': acc_record,
            #     }
            #     if not os.path.isdir('checkpoint'):
            #         os.mkdir('checkpoint')
            #     torch.save(state, './checkpoint/ckpt' + names + '.t7')
            #     best_acc = correct
