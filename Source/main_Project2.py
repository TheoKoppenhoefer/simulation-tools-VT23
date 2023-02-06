import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import IDA
from squeezer import Seven_bar_mechanism
import matplotlib.pyplot as mpl
import math


def run_seven_bar_problem(with_plots=True):
    """
    """
    tfinal = 0.03  # Specify the final time
    mod = Seven_bar_mechanism()

    # Define an explicit solver
    sim = IDA(mod)


    # Set the parameters
    atol = 1E-6*np.ones((20,))
    atol[14:20] = 1E5
    sim.atol = atol
    # Remove lambdas from the error test
    algvar = np.zeros((20,))
    algvar[14:20] = 1
    sim.algvar = algvar
    sim.suppress_alg = True

    # Simulate
    t, y, yd = sim.simulate(tfinal)


    # Plot
    if with_plots:
        # do some plotting
        var_labels = [r'$\beta$', r'$\theta$', r'$\gamma$', r'$\phi$', r'$\delta$', r'$\omega$', r'$\epsilon$',
                      r'$\dot{\beta}$', r'$\dot{\theta}$', r'$\dot{\phi}$', r'$\dot{\delta}$', r'$\dot{\omega}$',
                      r'$\dot{\omega}$', r'$\dot{\epsilon}$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                      r'$\lambda_4$', r'$\lambda_5$', r'$\lambda_6$', r'$\lambda_7$']
        for i in range(7):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[:7])
        mpl.figure()
        for i in range(7,14):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[7:14])
        mpl.figure()
        for i in range(14,20):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[14:])
        mpl.show()
    return mod, sim



if __name__ == '__main__':
    run_seven_bar_problem()
