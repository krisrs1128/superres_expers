import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import io
import torch
import torch.utils
from data import Combined


def medians(x):
    return torch.median(torch.stack(lr), dim=0)[0]


def trimmed_means(x, keep_mask, prop=0.1):
    x_hat = stats.trim_mean(torch.stack(lr), prop)
    return torch.from_numpy(x_hat)

def median_model(x, f=3):
    pred = medians(x)[0].numpy().astype("float32")
    pred = cv2.resize(pred, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    return pred.astype("int32")

data_dir = "/Users/krissankaran/Desktop/super-res/superres_data/train/NIR/"
combined = Combined(data_dir)
lr, hr, qm = combined[10]
loader = torch.utils.data.DataLoader(combined)

i = 0
ids = combined.high_res.ids
for lr, hr, qm in combined:
    pred = median_model(lr)
    io.imsave("/Users/krissankaran/Desktop/train_preds/" + ids[i] + "_pred.png", pred)
    io.imsave("/Users/krissankaran/Desktop/train_preds/" + ids[i] + "_truth.png", hr[0])
    i += 1

data_dir = "/Users/krissankaran/Desktop/super-res/superres_data/test/NIR/"
combined = Combined(data_dir)
lr, hr, qm = combined[10]
loader = torch.utils.data.DataLoader(combined)

i = 0
ids = combined.low_res.ids
for lr, hr, qm in combined:
    pred = median_model(lr)
    io.imsave("/Users/krissankaran/Desktop/test_preds/" + ids[i] + "_pred.png", pred)
    i += 1
