# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn
from ops.transforms import *
from torch.nn.init import normal_, constant_
import torch.nn.functional as F


import torchvision.models as models
import torch.nn as nn

class TSN(nn.Module):
    def __init__(self, num_class, modality='RGB', base_model='resnet50', pretrain='imagenet'):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_class = num_class
        self.pretrain = pretrain

        if 'resnet' in base_model:
            self.base_model = getattr(models, base_model)(pretrained=(self.pretrain == 'imagenet'))
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_class)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        return self.base_model(input)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if flip:
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(self.input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.input_mean, self.input_std)
            ])
        else:
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.input_size),
                torchvision.transforms.CenterCrop(self.input_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.input_mean, self.input_std)
        ])