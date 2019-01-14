#!/usr/bin/env python

"""
Toy version of the Proba-V dataset
"""
from torch.utils.data import Dataset
from PIL import Image
import os.path
import numpy as np
import glob
import torch
from torchvision import transforms


class ProbaBase(Dataset):
    """
    Shared Paths for Proba Data
    """
    def __init__(self, data_dir):
        super(ProbaBase, self).__init__()
        self.data_dir = data_dir
        self.imgsets = [os.path.join(data_dir, s) for s in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.imgsets)

    def __getitem__(self, index):
        return

    def getitem_wrapper(self, index, prefix="LR"):
        paths = glob.glob(os.path.join(self.imgsets[index], prefix + "*.png"))
        result = [transforms.ToTensor()(Image.open(s)) for s in paths]
        if len(result):
            return result[0]

        return result


class QualityMaps(ProbaBase):
    """
    Quality Maps for Low Res Data
    """
    def __init__(self, data_dir):
        super(QualityMaps, self).__init__(data_dir)

    def __getitem__(self, index):
        return self.getitem_wrapper(index, "QM")


class LowRes(ProbaBase):
    """
    List of Low Res Views of a Site
    """
    def __init__(self, data_dir):
        super(LowRes, self).__init__(data_dir)

    def __getitem__(self, index):
        return self.getitem_wrapper(index, "LR")


class HighRes(ProbaBase):
    """
    High Res Image for a Site
    """
    def __init__(self, data_dir):
        super(HighRes, self).__init__(data_dir)

    def __getitem__(self, index):
        return self.getitem_wrapper(index, "HR")


class Combined(ProbaBase):
    """
    Combined Proba-V Samples

    Examples
    --------
    data_dir = "/Users/krissankaran/Desktop/super-res/superres_data/train/NIR/"
    combined = Combined(data_dir)
    lr, hr, qm = combined[10]
    """
    def __init__(self, data_dir):
        super(Combined, self).__init__(data_dir)
        self.low_res = LowRes(data_dir)
        self.high_res = HighRes(data_dir)
        self.qm = QualityMaps(data_dir)

    def __getitem__(self, index):
        return self.low_res[index], self.high_res[index], self.qm[index]

