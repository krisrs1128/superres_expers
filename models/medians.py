from scipy import stats
from skimage import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.utils
from data import Combined


def medians(x):
    return torch.median(torch.stack(x), dim=0)[0]


def trimmed_means(x, keep_mask, prop=0.1):
    x_hat = stats.trim_mean(torch.stack(lr), prop)
    return torch.from_numpy(x_hat)


def median_model(x, f=3):
    pred = medians(x)[0].numpy().astype("float32")
    pred = cv2.resize(pred, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return pred.astype("int32")


def save_preds(output_path, combined):
    ids = combined.high_res.ids
    for i, (lr, hr, qm) in enumerate(combined):

        # make predictions
        pred = median_model(lr)
        io.imsave(os.path.join(output_path, ids[i] + "_pred.png"), pred)
        if len(hr) > 0:
            io.imsave(os.path.join(output_path, ids[i] + "_truth.png"), hr[0])


def save_wrapper(base, data_subdir, output_subdir):
    data_dir = base + data_subdir
    combined = Combined(data_dir)

    # create directory if doesn't exist
    if os.path.exists(base + output_subdir):
        shutil.rmtree(base + output_subdir)
    os.mkdir(base + output_subdir)

    save_preds(base + output_subdir, combined)


base = "/Users/krissankaran/Desktop/super-res/superres_data/"
save_wrapper(base, "train/RED/", "pred_red_train")
save_wrapper(base, "train/NIR/", "pred_nir_train")
save_wrapper(base, "test/RED/", "pred_red_test")
save_wrapper(base, "test/NIR/", "pred_nir_test")
