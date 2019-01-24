#!/usr/bin/env python
"""
Simulate lines for the superresolution experiment

"""
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.special import comb


###############################################################################
## Functions for drawing (high-res) line segments and bezier curves
###############################################################################

def curves(n=100, K=3, grid_len=100):
    """
    Collection of Bezier Curves

    :param n: The number of curves to draw
    :param K: The number of corners in each curve
    :param grid_len: The number of points in each curve
    :return x, corners: A tuple containing
      - x: A list of numpy arrays, each of which is grid_len x 2, containing
        the bezier curves
      - corners: A list of numpy arrays, each of which is K x 2, containing the
        control points that define the bezier curves
    """
    corners = []
    x = []
    for i in range(n):
        ci = np.random.normal(size=(K, 2))
        corners.append(ci)

        xi = bezier_curve(ci, grid_len)
        x.append(xi)

    return x, corners

def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
    I got this from: https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy

    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1],
                [2,3],
                [4,5], ..[Xn, Yn] ]
    nTimes is the number of time steps, defaults to 1000

    See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return np.flip(np.vstack((xvals, yvals)).T, axis=0)


def segments_(K=4, grid_len=100):
    """
    Piecewise Linear Curve (living in 2D)

    :param K: The number of corners defining the piecewise linear curve.
    :param grid_len: The number of points in the returned curve
    :return array, corners: A tuple containing,
      - arrays: An grid_len x 2 numpy array giving the values of the piecewise
        linear curve.
      - corners: The corners that define the piecewise linear curve
    """
    # K is the number of corners
    corners = np.random.normal(size=(K, 2))
    arrays = []
    for k in range(K - 1):
        edge = interpolate(corners[k, :], corners[k + 1, ], grid_len // (K - 1) + 1)
        arrays.append(np.array(edge))

    arrays = np.vstack(arrays)[:grid_len, :]
    arrays = (arrays - arrays.mean(0)) / arrays.std(0)
    return arrays, corners


def segments(n=100, K=4, grid_len=100):
    """
    Collection Piecewise Linear Curves (living in 2D)

    This just wraps segments_ over each of the n samples.

    :param n: The number of separate piecewise linear curves to return
    :param K: The number of corners defining the piecewise linear curve.
    :param grid_len: The number of points in the returned curve
    :return x, corners: A tuple containing
      - x: A length n list of arrays, each of which is grid_len x 2,
        representing the piecewise linear curves.
      - corners: A lengt h list of the K x 2 corners used to define each of the
        piecewise linear curves.
    """
    corners = []
    x = []
    for i in range(n):
        xi, ci = segments_(K, grid_len)
        x.append(xi)
        corners.append(ci)

    return x, corners


def interpolate(x0, x1, grid_len=100):
    """
    Uniformly spaced sequence from x0 to x1
    """
    array = []
    for i in range(grid_len):
        array.append((i / grid_len) * x0 + (1 - i / grid_len) * x1)

    return array

###############################################################################
## Approaches to downsampling and adding noise to the original high resolution
## sequences
###############################################################################

def downsample_(x, f, n_views=4):
    """
    Downsample according to an indexing function

    :param x: A numpy array that we want to downsample.
    :param f: A function that, when given x, returns a collection of indices in
      x to select.
    :param n_views: An integer specifying the number of downsampled versions of
      x to define.
    """
    x_down = []
    for i in range(n_views):
        x_down.append(x[f(x)])

    return x_down


def downsample_fun(size, sam_type):
    """
    Choose betewen downsampling approaches

    :param size: The size of the downsampled sequence
    :param sam_type: The approach to use for downsampling
    :return f: A function to use for downsampling (via the downsample_
      function)
    """
    if sam_type == "grid":
        f = lambda x: np.array(range(0, len(x), len(x) // size))[:size]
    elif sam_type == "random":
        f = lambda x: np.random.choice(len(x), size, replace=False)
    else:
        ArgumentError("Unrecognized sam_type for `downsample`")

    return f


def downsample(x, n_views=3, size=10, sam_type="grid"):
    """
    Wrap downsample_ across many sites

    :param x: The output of curves(), for example. A list of numpy arrays to
      downsample.
    :param n_views: An integer specifying the number of downsampled versions of
      x to define.
    :param size: The size of the downsampled sequence
    :param sam_type: The approach to use for downsampling
    :return x_low: A list of numpy arrays giving the downsampled views.
    """
    x_low = []
    for i in range(len(x)):
        f = downsample_fun(size, sam_type)
        x_low.append(downsample_(x[i], f, n_views))

    return x_low


def nested_gammas(n, k, l):
    # n sites, k views, each of length l
    return [[np.random.gamma(10, 0.005, size=(l, 2))] * k] * n

def noise(x, sigmas):
    """
    Add Gaussian Noise to Sites

    :param x: The output of curves(), for example. A list of numpy arrays to
      add noise to.
    :param sigmas: A list of list of numpy arrays, giving the amount of noise
        for the v^th view in the i^th site.
    :return x: The noisified version of the input x.

    Example
    -------
    x0, corners = curves(3)
    x_low = downsample(x0, size=50)
    x = noise(x_low, nested_gammas(3, 4, 50))

    for i in range(len(x_low)):
        for k in range(3):
            plt.scatter(x_low[i][k][:, 0], x_low[i][k][:, 1])
    """
    for i in range(len(x)):
        for v in range(len(x[i])):
            x[i][v] += sigmas[i][v] * np.random.normal(size=x[i][v].shape)
    return x

###############################################################################
## Dataset classes to use for actual experimentation
###############################################################################

def curves_wrapper(n_sites, K, n_views, hr_size, lr_size, sigmas, f=curves):
    """
    Wrapper for Initializing Curves Dataset

    :param n_sites: The number of curves to return.
    :param K: The number of corners in each curve
    :param n_views: An integer specifying the number of downsampled versions of
      each site to define.
    :param sigmas: A list of list of numpy arrays, giving the amount of noise
    :param hr_size: The size of the high resolution sequences
    :param lr_size: The size of the downsampled sequences
    :param f: A function to use for indexing. See downsample_ for examples.
    :return x_hr, x_lr: A tuple containing,
      - x_hr: A length n_sites list of numpy arrays giving coordinates of the
         high resolution curves.
      - x_lr: A length n_sites list of lists (of length n_views) of numpy arrays,
         giving coordinates of the low resolution views for each site.
    """
    if not sigmas:
        sigmas = nested_gammas(n_sites, n_views, lr_size)

    x_hr, _ = f(n_sites, K, hr_size)
    x_down = downsample(x_hr, n_views, lr_size)
    x_lr = noise(x_down, sigmas)

    # tensorify
    for i in range(len(x_hr)):
        x_hr[i] = torch.tensor(x_hr[i].copy()).float()
        for k in range(len(x_lr[i])):
            x_lr[i][k] = torch.tensor(x_lr[i][k].copy()).float()

    return x_hr, x_lr


class Curves(Dataset):
    """
    Dataset Class with high and low-res Curves

    Example
    -------
    ds = Curves()

    for i in range(len(ds)):
        plt.scatter(ds[i][0][:, 0], ds[i][0][:, 1], s=0.1, cmap=i) # low res
        for v in range(len(ds[i])):
            plt.scatter(ds[i][1][v][:, 0], ds[i][1][v][:, 1], s=0.2, cmap=i) # high res
    """
    def __init__(self, n_sites=5, K=3, n_views=3, hr_size=100, lr_size=25, sigmas=None):
        super(Curves, self).__init__()
        x_hr, x = curves_wrapper(n_sites, K, n_views, hr_size, lr_size, sigmas, segments)
        self.n_sites = n_sites
        self.n_views = n_views
        self.x_hr = x_hr
        self.x = x

    def __len__(self):
        return self.n_sites

    def __getitem__(self, idx):
        return self.x_hr[idx], self.x[idx]


class CurvesUnwrapped(Dataset):
    """
    Analog of Curves() but with each view indexed separately
    """
    def __init__(self, n_sites=5, K=3, n_views=3, hr_size=100, lr_size=25, sigmas=None):
        super(CurvesUnwrapped, self).__init__()
        x_hr, x = curves_wrapper(n_sites, K, n_views, hr_size, lr_size, sigmas, segments)
        self.n_sites = n_sites
        self.n_views = n_views
        self.x_hr = x_hr
        self.x = x

    def __len__(self):
        return self.n_sites * self.n_views

    def __getitem__(self, idx):
        site_idx = idx // self.n_views
        view_idx = idx % self.n_views
        return self.x_hr[site_idx], self.x[site_idx][view_idx]

