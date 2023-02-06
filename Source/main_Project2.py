import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import IDA
from squeezer import Seven_bar_mechanism
from squeezer2 import Seven_bar_mechanism_indx2
from squeezer1 import Seven_bar_mechanism_indx1
import matplotlib.pyplot as mpl
import math


def run_seven_bar_problem(with_plots=True, problem_index=3, atol_v=1E5, atol_lambda=1E5,
                          algvar_v=False, algvar_lambda=False):
    """
    """
    tfinal = 0.03  # Specify the final time

    if problem_index == 3:
        mod = Seven_bar_mechanism()
    elif problem_index == 2:
        mod = Seven_bar_mechanism_indx2()
    else:
        mod = Seven_bar_mechanism_indx1()

    # Define an implicit solver
    sim = IDA(mod)

    # Set the parameters
    atol = 1E-6*np.ones((20,))
    atol[7:14] = atol_v
    atol[14:20] = atol_lambda
    sim.atol = atol
    algvar = np.ones((20,))
    algvar[7:14] = algvar_v
    algvar[14:20] = algvar_lambda
    sim.algvar = algvar
    sim.suppress_alg = True

    # Simulate
    t, y, yd = sim.simulate(tfinal)


    # Plot
    if with_plots:
        # do some plotting
        var_labels = [r'$\beta$', r'$\Theta$', r'$\gamma$', r'$\phi$', r'$\delta$', r'$\Omega', r'$\epsilon$',
                      r'$\dot{\beta}$', r'$\dot{\Theta}$', r'$\dot{\phi}$', r'$\dot{\delta}$', r'$\dot{\omega}$',
                      r'$\dot{\Omega}$', r'$\dot{\epsilon}$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
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
