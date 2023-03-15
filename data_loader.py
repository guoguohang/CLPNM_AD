import math

import scipy.io
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, x_data, y_data, seed=0, args=None):
        self.x_data = x_data
        self.y_data = y_data
        self.seed = seed
        self.args = args

    def __getitem__(self, index):
        random_idx = np.random.randint(0, len(self))
        random_sample = self.x_data[random_idx]
        sample = self.x_data[index]
        label = self.y_data[index]
        sample = torch.tensor(sample)
        random_sample = torch.tensor(random_sample)

        corruption_mask = torch.zeros_like(sample, dtype=torch.bool)
        corruption_idx = torch.randperm(self.args.input_dim)[: self.args.corruption]
        corruption_mask[corruption_idx] = True
        random_sample = torch.where(corruption_mask, random_sample, sample)

        return sample, label, index, random_sample

    def __len__(self):
        return len(self.x_data)

    def to_dataframe(self):
        return pd.DataFrame(self.x_data, columns=self.columns)

    @property
    def shape(self):
        return self.x_data.shape


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.x_data.shape[0]


class DatasetSplit:
    def __init__(self, dataset_name=None, c_percent=0, seed=0):
        super(DatasetSplit, self).__init__()
        self.dataset_name = dataset_name
        self.c_percent = c_percent
        self.seed = seed

    def get_dataset(self):
        if self.dataset_name == 'micius':
            return self.micius_train_test_split(c_percent=self.c_percent)
        else:
            return self.mat_train_test_split(self.dataset_name, c_percent=self.c_percent)

    def mat_train_test_split(self, dataset_name, c_percent=0):
        dataset_path = os.path.join('data', dataset_name)
        data = scipy.io.loadmat(dataset_path, appendmat=True)
        samples = data['X']
        labels = ((data['y']).astype(np.int32)).reshape(-1)
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        return self.train_test_split(inliers, outliers, c_percent)

    def micius_train_test_split(self, c_percent=0):
        data = pd.read_csv("data/micius.csv")
        samples = data.drop(['label'], axis=1).values
        labels = data.loc[:, 'label'].values
        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        return self.train_test_split(inliers, outliers, c_percent)

    def train_test_split(self, inliers, outliers, c_percent=0):
        num_split = len(inliers) // 2
        np.random.seed(self.seed)
        randIdx = np.random.permutation(len(inliers))
        train_data = inliers[randIdx[:num_split]]
        if c_percent != 0:
            n_contaminated = math.ceil((c_percent / 100) * len(outliers))
            rpc = np.random.permutation(n_contaminated)
            train_data = np.concatenate([train_data, outliers[rpc]])
        train_label = np.zeros(len(train_data))
        test_data = np.concatenate([inliers[randIdx[num_split:]], outliers], 0)
        test_label = np.zeros(test_data.shape[0])
        test_label[-len(outliers):] = [1]*len(outliers)
        scaler = StandardScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, train_label, test_data, test_label

