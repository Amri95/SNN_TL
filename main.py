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
import dataloader
import os
import time
from spiking_model import*
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
names = 'transfer_spiking_model'
ROOT_PATH = './data/Office31'
SOURCE_NAME = 'amazon'
TARGET_NAME = 'webcam'

source_loader = dataloader.load_training(ROOT_PATH, SOURCE_NAME, batch_size)
target_train_loader = dataloader.load_training(ROOT_PATH, TARGET_NAME, batch_size)
target_test_loader = dataloader.load_testing(ROOT_PATH, TARGET_NAME, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path =  './data/raw/' # '/home/zhanqiugang/deep_snn/test/test_files/NMNISTsmall/' #todo: input your data path
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

# snn = torch.nn.DataParallel(SCNN(), device_ids=[0, 1, 2, 3])
snn = OldSCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    correct = 0
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        t0 = time.time()
        outputs = snn(images)
        # print(outputs)
        # print("forward time: ", time.time() - t0)
        labels_ = torch.zeros(batch_size, num_classes).scatter_(1, labels.view(-1, 1), 1)
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels).sum().item())
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        t0 = time.time()
        # print('before backward ---------------------------------------')
        # print(snn.h4_sumspike.grad)
        loss.backward()  # retain_graph=True)
        # print('after backward ---------------------------------------')
        # print(snn.h4_sumspike.grad)
        # print("backward time: ", time.time() - t0)
        optimizer.step()
        if i % 10 == 4:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, len(source_loader) * batch_size,
                100. * i / len(source_loader), loss))
        # if (i+1)%100 == 0:
        #      print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
        #             %(epoch+1, num_epochs, i+1, len(source_loader), running_loss))
        #      running_loss = 0
        #      print('Time elasped:', time.time()-start_time)

    running_loss /= len(train_loader)
    acc_train = float(correct) * 100. / (len(train_loader.dataset) * batch_size)

    print('{} set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        SOURCE_NAME, running_loss, correct, len(train_loader.dataset), acc_train))
    print('Time elasped:', time.time() - start_time)

    test_loss = 0
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(target_train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size, num_classes).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            test_loss += loss
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                # print(batch_idx, len(source_loader),' Acc: %.5f' % acc)

    test_loss /= len(target_train_loader)
    print('{} set: Average classification loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        TARGET_NAME, test_loss, correct, len(target_train_loader.dataset),
        100. * correct / len(target_train_loader.dataset)))

    # print('Iters:', epoch,'\n\n\n')
    # print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 5 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + names + '.t7')
        best_acc = acc