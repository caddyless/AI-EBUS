import os
import torch
import torchvision.transforms as transforms
from datamanager.folder2lmdb import ImageFolderLMDB


def data_loader(batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join('/home/data/imagenet_lmdb','ILSVRC-%s.lmdb'%'train')
    valdir = os.path.join('/home/data/imagenet_lmdb','ILSVRC-%s.lmdb'%'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    #traindir = os.path.join(root,'train')
    #train_dataset = datasets.ImageFolder(traindir,train_transform)
    #valdir = os.path.join(root,'val')
    #val_dataset = datasets.ImageFolder(valdir,val_transform)

    train_dataset=ImageFolderLMDB(traindir, train_transform)
    val_dataset=ImageFolderLMDB(valdir, val_transform)
    #print(train_dataset.__getitem__(1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
