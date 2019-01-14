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
        if len(result) == 1:
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


def subindex(shape, grid_num, x_ix, y_ix):
    size = shape[1] / grid_num
    y_range = int(size * y_ix), int(size * y_ix + size)
    x_range = int(size * x_ix), int(size * x_ix + size)
    return y_range, x_range

class ProbaPatches(Combined):

    def __init__(self, data_dir):
        super(ProbaPatches, self).__init__(data_dir)
        self.grid_num = 8

    def __len__(self):
        return 64 * len(self.hr)

    def __getitem__(self, index):
        im_index = index // (self.grid_num ** 2)
        patch_ix = index % self.grid_num
        x_ix = patch_ix % self.grid_num
        y_ix = patch_ix // self.grid_num

        hr_range = subindex(self.high_res[0].shape, self.grid_num, x_ix, y_ix)
        lr_range = subindex(self.low_res[0][0].shape, self.grid_num, x_ix, y_ix)
        high_res = self.high_res[im_index][:, hr_range[0][0]:hr_range[0][1], hr_range[1][0]:hr_range[1][1]]
        print(lr_range)
        low_res = [s[:, lr_range[0][0]:lr_range[0][1], lr_range[1][0]:lr_range[1][1]] for s in self.low_res[im_index]]
        qm = [s[:, lr_range[0][0]:lr_range[0][1], lr_range[1][0]:lr_range[1][1]] for s in self.qm[im_index]]

        return low_res, high_res, qm
