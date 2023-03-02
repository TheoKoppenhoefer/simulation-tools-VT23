# /usr/bin/env python

import numpy as np
from assimulo.problem import Explicit_Problem
from scipy.sparse.linalg import spsolve


class Explicit_Problem_2nd(Explicit_Problem):
    def __init__(self, M, C, K, u0, ud0, t0, f, **params):
        self.M = M
        self.C = C
        self.K = K
        self.u0 = u0
        self.ud0 = ud0
        self.t0 = t0
        self.f = f
        Explicit_Problem.__init__(self, self.rhs, np.concatenate((u0, ud0)), t0, **params)

    def rhs(self, t, y):
        n = y.size // 2
        u, v = y[:n], y[n:]
        M = self.M
        K = self.K
        C = self.C
        f = self.f
        return np.concatenate((v, f(t) - spsolve(M, K@ u+ C@ v)))
