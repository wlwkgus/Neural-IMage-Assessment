#!/usr/bin/env python
# -*- coding: utf-8

from __future__ import absolute_import

import os

import pandas as pd
from PIL import Image
from PIL import ImageFile
from glob import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils import data

not_included = [
    953619,
    953958,
    954184,
    954113,
    953980,
    954175,
    953349,
    953645,
    953897,
    953841,
    310261,
    848725,
    444892,
    567829,
    398594,
    638163,
    397289,
    104855,
    11066,
    148477,
    52365,
    430454,
]


def split_dataset(ava_path):
    train_frac = 0.95
    val_frac = 0.05
    test_frac = 0.0

    total = pd.read_csv(ava_path, delimiter=' ', header=None)
    total = total.loc[:, 1:11]
    # normalize total
    total[12] = total.iloc[:, 2:].sum(axis=1)
    for i in range(2, 12):
        total[i] = total[i] / total[12]
    total = total.loc[:, 1:11]
    total = total[~total[1].isin(not_included)]
    df_train = total.sample(frac=train_frac)
    total.drop(df_train.index, inplace=True)
    df_val = total.sample(frac=(val_frac / (val_frac + test_frac)))
    total.drop(df_val.index, inplace=True)
    df_test = total
    df_train.to_csv('ann_train.csv', header=None)
    df_val.to_csv('ann_val.csv', header=None)
    df_test.to_csv('ann_test.csv', header=None)


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None, usecols=range(1, 12))
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


class TestDataset(data.Dataset):
    """
    Custom dataset.
    Scan all jpg files in given image directory.

    Args:
        image_dir: image_dir
    """

    def __init__(self, image_dir, root_dir, transform=None):
        self.image_dir = image_dir
        self.root_dir = root_dir
        self.transform = transform
        self.full_image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_names = glob(self.full_image_path + '/*.jpeg')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(img_name)
        sample = dict()
        sample['image'] = image
        sample['image_name'] = img_name
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
