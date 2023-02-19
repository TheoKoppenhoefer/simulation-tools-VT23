#/usr/bin/env python

import numpy as np
from assimulo.problem import Explicit_Problem

class Explicit_Problem_2nd(Explicit_Problem):
    def __init__(self, M, C, K, u0, ud0, t0, f):
        self.M = M
        self.C = C
        self.K = M
        self.u0 = u0
        self.ud0 = ud0
        self.t0 = t0
        self.f = f
        Explicit_Problem.__init__(self, self.rhs, y0=np.concatenate((u0, ud0)), t0=t0)

    def rhs(self, t, y):
        n = y.size / 2
        u, v = y[:n], y[n:]
        return np.concatenate((v, self.f(t) - np.linalg.solve(self.M, np.dot(K, u) \
                                - np.linalg.solve(self.M, np.dot(C, v))
