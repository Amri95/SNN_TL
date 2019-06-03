
import os
import torch

from torchvision import datasets, transforms

def load_training(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],  # [0.485, 0.456, 0.406], [0.5, 0.5, 0.5],
                              std=[1, 1, 1])  # [0.229, 0.224, 0.225]) [0.5, 0.5, 0.5])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

def load_testing(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([227, 227]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],  # [0.485, 0.456, 0.406], [0.5, 0.5, 0.5],
                              std=[1, 1, 1])  # [0.229, 0.224, 0.225]) [0.5, 0.5, 0.5])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return test_loader

def load_data(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],  # [0.485, 0.456, 0.406], [0.5, 0.5, 0.5],
                              std=[1, 1, 1])  # [0.229, 0.224, 0.225]) [0.5, 0.5, 0.5])
         ]
    )
    data = datasets.ImageFolder(root=os.path.join(root_path, directory, 'images'), transform=transform)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader
