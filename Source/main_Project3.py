import numpy as np
from assimulo.problem import Explicit_Problem
from Explicit_Problem_2nd import Explicit_Problem_2nd
from HHT import HHT, Newmark_explicit, Newmark_implicit
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl
import math
from main_Project1 import f_pendulum, plot_pendulum, pendulum_energies


def run_elastic_pendulum_problem(with_plots=True, k=1., atol=1E-6, rtol=1E-6, maxord=5, discr='BDF'):
    """
    """
    rhs = lambda t,y: f_pendulum(t, y, k)

    def get_udd_n(t, u_n):
        return rhs(0, np.concatenate((u_n,np.zeros(2))))[2:]

    def get_udd0(t, u_0, ud_0):
        return rhs(0, np.concatenate((u_0,np.zeros(2))))[2:]

    y0 = np.array([1.1, 0, 0, 0])
    t0 = 0.0
    tfinal = 10.0  # Specify the final time
    # Define an Assimulo problem
    u0 = np.array([1.1, 0])
    ud0 = np.zeros(2)

    mod = Explicit_Problem_2nd(None, None, None, u0, ud0, t0, None, name=r'Elastic Pendulum Problem')
    mod.rhs = rhs

    # Define an explicit solver
    sim = Newmark_explicit(mod)

    # Sets the parameters
    sim.get_udd0 = get_udd0
    sim.get_udd_n = get_udd_n

    # Simulate
    t, y = sim.simulate(tfinal)

    pot_energy, kin_energy, elast_energy, total_energy, stability_index = pendulum_energies(y, k)

    # Plot
    if with_plots:
        plot_pendulum(t, y, pot_energy, kin_energy, elast_energy, total_energy)
    return mod, sim, stability_index

if __name__ == '__main__':
    run_elastic_pendulum_problem()
    # mpl.show()

    # TODO: generate experiments testing elastodyn.py
