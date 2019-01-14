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
    def __init__(self, data_dir, prefix="LR"):
        super(ProbaBase, self).__init__()
        self.data_dir = data_dir
        imgsets = [os.path.join(data_dir, s) for s in os.listdir(self.data_dir)][:25]
        self.images = []

        sites = [glob.glob(os.path.join(s, prefix + "*.png")) for s in imgsets]
        for views in sites:
            x = [transforms.ToTensor()(Image.open(s)) for s in views]
            if len(x) == 1:
                x = x[0]

            self.images.append(x)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


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
        super(ProbaBase, self).__init__()
        self.low_res = ProbaBase(data_dir, "LR")
        self.high_res = ProbaBase(data_dir, "HR")
        self.qm = ProbaBase(data_dir, "QM")

    def __getitem__(self, index):
        return self.low_res[index], self.high_res[index], self.qm[index]


class ProbaPatches(Combined):
    """
    16 x 16 (and 48 x 48) patches in Proba-V

    The 16 x 16 patches are low res patches, the 48 x 48 are high res. The
    patches are returned consecutively across images, starting at the top left
    and marching down and to the right.

    Examples
    --------
    data_dir = "/Users/krissankaran/Desktop/super-res/superres_data/train/NIR/"
    patches = ProbaPatches(data_dir)
    patches[1]
    test = torch.utils.data.DataLoader(patches)
    next(iter(test))
    """
    def __init__(self, data_dir):
        super(ProbaPatches, self).__init__(data_dir)

    def __len__(self):
        return 64 * len(self.high_res)

    def __getitem__(self, index):
        im_index = index // (8 ** 2)
        patch_ix = index % 8
        x_ix = patch_ix % 8
        y_ix = patch_ix // 8

        hr = self.high_res[im_index].unfold(1, 8, 8).unfold(2, 8, 8)
        lr = [s.unfold(1, 8, 8).unfold(2, 8, 8) for s in self.low_res[im_index]]
        lr = [s[:, :, :, x_ix, y_ix] for s in lr]
        qm = [s.unfold(1, 8, 8).unfold(2, 8, 8) for s in self.qm[im_index]]
        qm = [s[:, :, :, x_ix, y_ix] for s in qm]
        return lr, hr[:, :, :, x_ix, y_ix], qm
