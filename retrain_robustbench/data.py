from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, KMNIST, FashionMNIST, ImageFolder
import os
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
import pandas as pd


class ImageNet1k(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        
        if split == "val":
            df_id_to_syn = pd.read_csv(os.path.join(root, f"LOC_{split}_solution.csv")).set_index("ImageId")
            df_id_to_syn["PredictionString"] = df_id_to_syn["PredictionString"].apply(lambda s: s.split(" ")[0])
        
        samples_dir = os.path.join(root, split)
        
        for root_dir, dirs, files in os.walk(samples_dir):
            for file in filter(lambda f: f.endswith(".JPEG"), files):
                sample_path = os.path.join(root_dir, file)
                image_id = file.split(".")[0]
                if split == "val":    
                    syn_id = df_id_to_syn.loc[image_id].values[0]
                elif split == "train":
                    syn_id = image_id.split("_")[0]
                
                class_id = self.syn_to_class[syn_id]
                self.samples.append(sample_path)
                self.targets.append(class_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNet1kData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 1000
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="train", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="val", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()



class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = CIFAR10(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        dataset = CIFAR10(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.50707516, 0.48654887, 0.44091784)
        self.std = (0.26733429, 0.25643846, 0.27615047)
        self.num_classes = 100
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = CIFAR100(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        dataset = CIFAR100(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CINIC10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers, part="all"):
        super().__init__()
        assert part in ["all", "imagenet", "cifar10"]
        self.part = part
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.47889522, 0.47227842, 0.43047404)  # from https://github.com/BayesWatch/cinic-10
        self.std = (0.24205776, 0.23828046, 0.25874835)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageFolder(root=os.path.join(self.root_dir, "train"), transform=transform, is_valid_file= \
                              lambda path: (self.part == "all") or \
                              (self.part == "imagenet" and not os.path.basename(path).startswith("cifar10-")) or \
                              (self.part == "cifar10" and os.path.basename(path).startswith("cifar10-")))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        dataset = ImageFolder(root=os.path.join(self.root_dir, "valid"), transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class TensorData(pl.LightningDataModule):
    def __init__(self, data_class, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = data_class

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class MNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(MNIST, root_dir, batch_size, num_workers)
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.num_classes = 10
        self.in_channels = 1


class KMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(KMNIST, root_dir, batch_size, num_workers)
        self.mean = (0.1918,)
        self.std = (0.3483,)
        self.num_classes = 49
        self.in_channels = 1


class FashionMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(FashionMNIST, root_dir, batch_size, num_workers)
        self.mean = (0.2860,)
        self.std = (0.3530,)
        self.num_classes = 10
        self.in_channels = 1


all_datasets = {
    "cifar10": CIFAR10Data,
    "cifar100": CIFAR100Data,
    "mnist": MNISTData,
    "kmnist": KMNISTData,
    "fashionmnist": FashionMNISTData,
    "cinic10": CINIC10Data,
    "imagenet1k": ImageNet1kData
}


def get_dataset(name):
    return all_datasets.get(name)
