#!/usr/bin/env python
"""
Simulate lines for the superresolution experiment

"""
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def segments_(K=4, grid_len=100):
    corners = np.random.normal(0, 1, (K, 2))
    arrays = []
    for k in range(K - 1):
        edge = interpolate(corners[k, :], corners[k + 1, ], grid_len // (K - 1) + 1)
        arrays.append(np.array(edge))

    arrays = np.vstack(arrays)[:grid_len, :]
    return arrays, corners


def segments(n=100, K=4, grid_len=100):
    corners = []
    x = []
    for i in range(n):
        xi, ci = segments_(K, grid_len)
        x.append(xi)
        corners.append(ci)

    return x, corners


def interpolate(x0, x1, grid_len=100):
    array = []
    for i in range(grid_len):
        array.append((i / grid_len) * x0 + (1 - i / grid_len) * x1)

    return array


def downsample_(x, size=10, n_views=4):
    x_down = []
    for i in range(n_views):
        z = np.random.choice(len(x), size, replace=False)
        x_down.append(x[z])

    return x_down


def downsample(x, n_views=3, size=10):
    x_low = []
    for i in range(len(x)):
        x_low.append(downsample_(x[i], size, n_views))

    return x_low


def noise(x, sigma=0.05):
    """

    Example
    -------
    x0, corners = segments(3)
    x_low = downsample(x0, 10)
    x = noise(x_low)

    for i in range(len(x_low)):
        for k in range(10):
            plt.scatter(x_low[i][k][:, 0], x_low[i][k][:, 1])
    """
    for i in range(len(x)):
        for k in range(len(x[i])):
            x[i][k] += np.random.normal(0, sigma, x[i][k].shape)
    return x


class Segments(Dataset):
    def __init__(self, n_samples=100, K=4, n_views=3, size=10, sigma=0.05):
        super(Segments, self).__init__()
        self.n_samples = n_samples
        self.n_views = n_views
        self.sigma = sigma
        x0, corners = segments(n_samples, K)
        x_down = downsample(x0, n_views, size)
        x = noise(x_down, sigma)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return x[idx]


class SegmentsUnwrapped(Dataset):
    def __init__(self, n_samples=100, K=4, n_views=3, size=10, sigma=0.05):
        super(SegmentsUnwrapped, self).__init__()
        self.n_samples = n_samples
        self.n_views = n_views
        self.sigma = sigma
        x0, corners = segments(n_samples, n_views)
        x_down = downsample(x0, n_views, size)

        # save variables of interest
        self.x0 = x0
        self.corners = corners
        self.x = noise(x_down, sigma)

    def __len__(self):
        return self.n_samples * self.n_views

    def __getitem__(self, idx):
        return self.x[idx // self.n_views][idx % self.n_views]

ds = SegmentsUnwrapped()
for i in range(100):
    plt.scatter(ds[i][:, 0], ds[i][:, 1])

