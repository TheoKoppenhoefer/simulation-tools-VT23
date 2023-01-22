

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl

y0=np.ones(4)
t0=0.0
k=1
def rhs(t,y):
    yd = np.zeros(y.size)
    yd[0:2] = y[2:4]
    norm_y = np.linalg.norm(y[0:2])
    lam = k*(norm_y-1)/norm_y
    yd[2] = -y[0]*lam
    yd[3] = -y[1]*lam-1
    return yd


if __name__ == '__main__':

    model = Explicit_Problem(rhs, y0, t0) #Create an Assimulo problem
    model.name = 'Linear Test ODE'

    sim = CVode(model)
    tfinal = 10.0        #Specify the final time
    t, y = sim.simulate(tfinal)
    sim.plot()
    mpl.show()

