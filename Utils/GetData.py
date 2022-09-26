import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy.io import loadmat
import h5py
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = "cuda:2" if torch.cuda.is_available() else "cpu"
device3 = "cuda:3" if torch.cuda.is_available() else "cpu"



class GetData(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, trainingpath, trainingfile, device):
        """用来训练和验证"""
        Y = h5py.File(f'{trainingpath}/{trainingfile[0]}')
        Y = np.transpose(Y[f"{trainingfile[0].split('.')[0]}"])
        self.echo = torch.tensor(Y, dtype=torch.float32, device=device)
        self.echo = self.echo/self.echo.max(dim=-1, keepdim=True).values

        X = h5py.File(f'{trainingpath}/{trainingfile[1]}.mat')
        X = np.transpose(X[f"{trainingfile[1].split('.')[0]}"])
        self.target = torch.tensor(X, dtype=torch.float32, device=device)
        self.device = device
    def __getitem__(self, item):
        data = {}
        data['echo'] = self.echo[item]
        data['target'] = self.target[item]
        return data

    def __len__(self):
        return len(self.echo)

class GetTestData(Dataset):
    """
    return data for dataloader
    """
    def __init__(self, testingpath, testingfile, device):
        """用来测试"""
        Y = h5py.File(f'{testingpath}/{testingfile[0]}')
        Y = np.transpose(Y[f"{testingfile[0].split('.')[0]}"])
        self.echo = torch.tensor(Y, dtype=torch.float32, device=device)

        self.echo = self.echo / self.echo.max(dim=-1, keepdim=True).values
        X = h5py.File(f'{testingpath}/{testingfile[1]}.mat')
        X = np.transpose(X[f"{testingfile[1].split('.')[0]}"])
        self.target = torch.tensor(X, dtype=torch.float32, device=device)

    def __getitem__(self, item):
        data = {}
        data['echo'] = self.echo[item]
        data['target'] = self.target[item]
        return data

    def __len__(self):
        return len(self.echo)
