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

n = 100
x = np.arange(n, dtype=np.float64)
a = ot.datasets.make_1D_gauss(n, m=20, s=5)
b = ot.datasets.make_1D_gauss(n, m=60, s=10)

M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

Ge = ot.emd_1d(x, x, a, b)
# print(Ge)

# Equivalent to
# G0 = ot.emd(a, b, M)

lambd = 1e-3
Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)
# print(Gs)

mse = ((Ge - Gs) ** 2).mean(axis=None)
print(mse)