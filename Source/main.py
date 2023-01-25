import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl


def run_elastic_pendulum_problem(with_plots=True, k=1, atol=1E-6, rtol=1E-6, maxord=5):
    """
    """

    # Define the rhs
    def rhs(t, y):
        yd = np.zeros(y.size)
        yd[0:2] = y[2:4]
        norm_y = np.linalg.norm(y[0:2])
        lam = k * (norm_y - 1) / norm_y
        yd[2] = -y[0] * lam
        yd[3] = -y[1] * lam - 1
        return yd

    y0 = np.array([1.1, 0, 0, 0])
    t0 = 0.0
    tfinal = 10.0  # Specify the final time
    # Define an Assimulo problem
    mod = Explicit_Problem(rhs, y0, t0, name=r'Elastic Pendulum Problem')

    # Define an explicit solver
    sim = CVode(mod)  # Create a CVode solver

    # Sets the parameters
    sim.atol = [atol]
    sim.rtol = rtol
    sim.maxord = maxord

    # Simulate
    t, y = sim.simulate(tfinal)

    # Plot
    if with_plots:
        for i in range(y.shape[1]):
            mpl.plot(t, y[:, i])
        mpl.legend([r'$y_1$', r'$y_2$', r'$\dot{y}_1$', r'$\dot{y}_2$'])
        mpl.show()
        mpl.plot(y[:, 0], y[:, 1])
        mpl.xlabel(r'$y_1$')
        mpl.ylabel(r'$y_2$')
        mpl.title('phase portrait')
        mpl.show()
    return mod, sim


if __name__ == '__main__':
    run_elastic_pendulum_problem()
