from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class AffectNet(data.Dataset):
    """Dataset class for the AffectNet dataset."""

    def __init__(self, image_dir, affectnet_emo_descr, transform, mode):
        """Initialize and preprocess the AffectNet dataset."""
        self.image_dir = image_dir
        self.affectnet_emo_descr = affectnet_emo_descr
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)


    def preprocess(self):
        """Preprocess the AffectNet emotional description file."""
        if self.affectnet_emo_descr == 'emotiw':
            lines_train = [line.rstrip() for line in open(os.path.join(self.image_dir, '_affectnet_train_emotiw.txt'), 'r')]
            for line in lines_train:
                filename, label = line.split()
                self.train_dataset.append([filename, int(label)])
            lines_test = [line.rstrip() for line in open(os.path.join(self.image_dir, '_affectnet_test_emotiw.txt'), 'r')]
            for line in lines_test:
                filename, label = line.split()
                self.test_dataset.append([filename, int(label)])
        else:
            lines_train = [line.rstrip() for line in open(os.path.join(self.image_dir, '_affectnet_train_va.txt'), 'r')]
            for line in lines_train:
                split = line.split()
                filename, v, a = split
                self.train_dataset.append([filename, [float(v), float(a)]])
            lines_test = [line.rstrip() for line in open(os.path.join(self.image_dir, '_affectnet_test_va.txt'), 'r')]
            for line in lines_test:
                split = line.split()
                filename, v, a = split
                self.test_dataset.append([filename, [float(v), float(a)]])

        print('Finished preprocessing the AffectNet dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == 'train':
            dataset = self.train_dataset
            filename, label = dataset[index]
            image = Image.open(os.path.join(self.image_dir, 'train', filename))
        else:
            dataset = self.test_dataset
            filename, label = dataset[index]
            image = Image.open(os.path.join(self.image_dir, 'validation', filename))
        return self.transform(image), torch.tensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', affectnet_emo_descr='emotiw', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
    elif dataset == 'AffectNet':
        dataset = AffectNet(image_dir, affectnet_emo_descr, transform, mode)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
