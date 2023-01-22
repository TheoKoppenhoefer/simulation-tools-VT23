

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl
import pylab as pl




def run_elastic_pendulum_problem(with_plots=True,k=1):
    """
    """

    # Define the rhs
    def rhs(t,y):
        yd = np.zeros(y.size)
        yd[0:2] = y[2:4]
        norm_y = np.linalg.norm(y[0:2])
        lam = k*(norm_y-1)/norm_y
        yd[2] = -y[0]*lam
        yd[3] = -y[1]*lam-1
        return yd

    y0 = np.ones(4)
    t0 = 0.0
    tfinal = 10.0        #Specify the final time
    # Define an Assimulo problem
    mod = Explicit_Problem(rhs, y0, t0, name=r'Elastic Pendulum Problem')

    # Define an explicit solver
    sim = CVode(mod)  # Create a CVode solver

    # Sets the parameters
    # sim.atol = [1e-4]  # Default 1e-6
    # sim.rtol = 1e-4  # Default 1e-6

    # Simulate
    t, y = sim.simulate(tfinal)

    # Plot
    if with_plots:
        pl.plot(t, y, color="b")
        pl.title(mod.name)
        pl.ylabel('y')
        pl.xlabel('Time')
        pl.show()
        sim.plot()
        mpl.show()
    return mod, sim

if __name__ == '__main__':
    run_elastic_pendulum_problem()

