
import os
import yaml
import torch

from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision.datasets.folder import *


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, source_root, target_root,
                 extensions=None,
                 transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        #         super(ImageFolder, self).__init__(self, source_root, target_root,
        #                  extensions=None,
        #                  transform=None, target_transform=None,
        #                  loader=default_loader, is_valid_file=None)
        self.source_root = source_root
        self.target_root = target_root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None

        source_classes, source_class_to_idx = self._find_classes(self.source_root)
        target_classes, target_class_to_idx = self._find_classes(self.target_root)

        source_samples = make_dataset(self.source_root, source_class_to_idx, self.extensions, self.is_valid_file)
        target_samples = make_dataset(self.target_root, target_class_to_idx, self.extensions, self.is_valid_file)

        if len(source_samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.source_root + "\n"
                                                                                        "Supported extensions are: " + ",".join(
                extensions)))
        if len(target_samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.target_root + "\n"
                                                                                        "Supported extensions are: " + ",".join(
                extensions)))

        self.source_classes = source_classes
        self.source_class_to_idx = source_class_to_idx
        self.source_samples = source_samples
        self.source_targets = [s[1] for s in source_samples]

        self.target_classes = source_classes
        self.target_class_to_idx = target_class_to_idx
        self.target_samples = target_samples
        self.target_targets = [s[1] for s in target_samples]

        self.source_merge_imgs = []
        self.target_merge_imgs = []
        self.source_merge_targets = []
        self.target_merge_targets = []

        for i in self.source_class_to_idx.values():

            source_start = self.source_targets.index(i)
            source_L = self.source_targets.count(i)
            source_idx = list(range(source_start, source_start + source_L))

            target_start = self.target_targets.index(i)
            target_L = self.target_targets.count(i)
            target_idx = list(range(target_start, target_start + target_L))

            if source_L > target_L:
                times = int(source_L / target_L)
                remainder = source_L % target_L
                target_idx *= times
                target_idx += target_idx[:remainder]

                self.source_merge_imgs += [self.source_samples[x] for x in source_idx]
                self.target_merge_imgs += [self.target_samples[x] for x in target_idx]

                self.source_merge_targets += [self.source_targets[x] for x in source_idx]
                self.target_merge_targets += [self.target_targets[x] for x in target_idx]
            elif source_L < target_L:
                times = int(target_L / source_L)
                remainder = target_L % source_L
                source_idx *= times
                source_idx += source_idx[:remainder]

                self.source_merge_imgs += [self.source_samples[x] for x in source_idx]
                self.target_merge_imgs += [self.target_samples[x] for x in target_idx]

                self.source_merge_targets += [self.source_targets[x] for x in source_idx]
                self.target_merge_targets += [self.target_targets[x] for x in target_idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        source_path, source_target = self.source_merge_imgs[index]
        target_path, _ = self.target_merge_imgs[index]

        source_sample = self.loader(source_path)
        target_sample = self.loader(target_path)

        if self.transform is not None:
            source_sample = self.transform(source_sample)
            target_sample = self.transform(target_sample)

        if self.target_transform is not None:
            source_target = self.target_transform(source_target)

        return source_sample, target_sample, source_target

    def __len__(self):
        return len(self.source_merge_imgs)


def load_merge_training(root_path, source_directory, target_directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],  # [0.485, 0.456, 0.406], [0.5, 0.5, 0.5], [0, 0, 0]
                              std=[1, 1, 1])  # [0.229, 0.224, 0.225]) [0.5, 0.5, 0.5]) [1, 1, 1])
         ]
    )
    data = ImageFolder(source_root=os.path.join(root_path, source_directory, 'images'),
                       target_root=os.path.join(root_path, target_directory, 'images'),
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader


def load_training(root_path, directory, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(227),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0, 0, 0],  # [0.485, 0.456, 0.406], [0.5, 0.5, 0.5], [0, 0, 0]
                              std=[1, 1, 1])   # [0.229, 0.224, 0.225]) [0.5, 0.5, 0.5]) [1, 1, 1])
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


# Consider dictionary for easier iteration and better scalability
class SlayerParams(object):
    def __init__(self, parameter_file_path):
        with open(parameter_file_path, 'r') as param_file:
            self.parameters = yaml.load(param_file)

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value
