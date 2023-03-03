import numpy as np
from assimulo.problem import Explicit_Problem
from Explicit_Problem_2nd import Explicit_Problem_2nd
from HHT import HHT, Newmark_explicit, Newmark_implicit
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl
import math
from main_Project1 import f_pendulum, plot_pendulum, pendulum_energies
from elastodyn import *


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
    run_beam_problem_HHT('HHT', 0, 0.5, 0.5, True, 8)
    # mpl.show()

    if False:
        # TODO: change this to compare solns
        
        # This plots comparisons of the index 1,2,3 formulations
        all_solns = []
        for i in range(4):
            _, _, soln = run_seven_bar_problem(False, i)
            if i <= 0:
                # Do some padding in the case of the explicit problem
                soln[1] = np.hstack((soln[1], np.zeros((len(soln[0]), 6))))
            all_solns.append(soln)

        t = np.linspace(0, 0.03, 500)
        all_solns_interp = np.zeros((4, t.size, 20))
        for i, soln in enumerate(all_solns):
            for j in range(20):
                all_solns_interp[i, :, j] = np.interp(t, soln[0], soln[1][:, j])

        # Plot soln
        plot_soln(all_solns[1][0], all_solns[1][1], savefig=True, plotnumber=510)
        plot_soln(all_solns[2][0], all_solns[2][1], savefig=True, plotnumber=513)
        plot_soln(all_solns[3][0], all_solns[3][1], savefig=True, plotnumber=516)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[0, :, :], savefig=True, plotnumber=520)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[1, :, :], savefig=True, plotnumber=530)
        plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[2, :, :], savefig=True, plotnumber=540)
        # mpl.show()

    if False:
        # This compares the index=1,2,3 formulations
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [2, 1E5, 1E5, False, False, True],
                       [3, 1E5, 1E5, False, False, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'{exp[0]}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=600, savefig=True, xlabel='index', figsize=(2,2))

    if False:
        # This tests the index=1 problem
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [1, 1E-6, 1E5, False, True, True],
                       [1, 1E-6, 1E5, True, False, True],
                       [1, 1E-6, 1E5, True, True, False],
                       [1, 1E-6, 1E-6, False, False, True]]


        # print(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='fancy_grid'))
        with open('../Plots/Tables/Overview_Index1Experiment.tex', 'w') as output:
            output.write(tabulate(experiments, headers=tab_headers, showindex='always', tablefmt='latex'))

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'{counter}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=700, savefig=True, xlabel='experiment', figsize=(2,2))

    if False:
        # This tests the index=2 problem
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[2, 1E-6, 1E5, False, False, True],
                       [2, 1E-6, 1E5, False, True, True],
                       [2, 1E-6, 1E5, False, True, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'prblm {counter}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=800, savefig=True, figsize=(2,2))
    # mpl.show()
