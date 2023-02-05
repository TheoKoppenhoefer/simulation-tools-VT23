import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import IDA
from squeezer import Seven_bar_mechanism
import matplotlib.pyplot as mpl
import math


def run_seven_bar_problem(with_plots=True):
    """
    """
    tfinal = 10.0  # Specify the final time
    mod = Seven_bar_mechanism()

    # Define an explicit solver
    sim = IDA(mod)

    # Simulate
    t, y = sim.simulate(tfinal)


    # Plot
    if with_plots:
        # do some plotting
        pass
    return mod, sim



if __name__ == '__main__':
    run_seven_bar_problem()
