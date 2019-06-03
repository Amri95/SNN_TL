import os
import torch

from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import scipy.misc

def load_training(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_testing(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([227, 227]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader

def load_data(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader

names = 'transfer_spiking_model'
ROOT_PATH = './data/Office31'
SOURCE_NAME = 'amazon'
TARGET_NAME = 'webcam'
batch_size = 1

source_loader, target_test_loader = load_data(ROOT_PATH, SOURCE_NAME, batch_size)
target_loader = load_training(ROOT_PATH, TARGET_NAME, batch_size)

# source_loader = dataloader.load_training(ROOT_PATH, SOURCE_NAME, batch_size)
# target_train_loader = dataloader.load_training(ROOT_PATH, TARGET_NAME, batch_size)
# target_test_loader = dataloader.load_testing(ROOT_PATH, TARGET_NAME, batch_size)

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

    source_data, source_label = source_data.cpu().detach().numpy(), source_label.cpu().detach().numpy()
    target_data = target_data.cpu().detach().numpy()

    print(source_data)
    print(source_label)
    print(target_data)

    im = Image.fromarray(source_data[0].swapaxes(1, 2, 0))
    im.save("source_data.jpeg")
    im = Image.fromarray(target_data[0].swapaxes(1, 2, 0))
    im.save("target_data.jpeg")

    break
