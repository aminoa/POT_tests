# -*- coding: utf-8 -*-
"""
======================================
Optimal Transport for 1D distributions
======================================

This example illustrates the computation of EMD and Sinkhorn transport plans
and their visualization.

"""

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss

n = 50  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n 
M = ot.dist(xs, xt)

Ge = ot.emd(a, b, M)
lambd = 1e-3
Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)
# print(Gs)

mse = ((Ge - Gs) ** 2).mean(axis=None)
print(mse)