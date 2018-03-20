#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import absolute_import

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data


def split_dataset(ava_path):
    train_frac = 0.7
    val_frac = 0.1
    test_frac = 0.2

    total = pd.read_csv(ava_path, delimiter=' ', header=None)
    total = total.loc[:, 1:11]
    df_train = total.sample(frac=train_frac)
    total.drop(df_train.index, inplace=True)
    df_val = total.sample(frac=(val_frac / (val_frac + test_frac)))
    total.drop(df_val.index, inplace=True)
    df_test = total
    df_train.to_csv('ann_train.csv')
    df_val.to_csv('ann_val.csv')
    df_test.to_csv('ann_test.csv')


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        annotations = self.annotations.iloc[idx, 1:].as_matrix()
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
