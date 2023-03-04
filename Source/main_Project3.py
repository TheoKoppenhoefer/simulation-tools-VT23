import numpy as np
from assimulo.problem import Explicit_Problem
from Explicit_Problem_2nd import Explicit_Problem_2nd
from HHT import HHT, Newmark_explicit, Newmark_implicit
from assimulo.solvers import ExplicitEuler, RungeKutta4
import matplotlib.pyplot as mpl
import math
from main_Project1 import f_pendulum, plot_pendulum, pendulum_energies
from elastodyn import *
from tabulate import tabulate
from main_Project1 import run_elastic_pendulum_problem


def run_elastic_pendulum_problem_newmark(with_plots=True, k=1., atol=1E-6, rtol=1E-6, maxord=5, discr='BDF'):
    """
    """
    rhs = lambda t,y: f_pendulum(t, y, k)

    def get_udd_n(t, u_n):
        return rhs(0, np.concatenate((u_n,np.zeros(2))))[2:]

    def get_udd0(t, u_0, ud_0):
        return rhs(0, np.concatenate((u_0,np.zeros(2))))[2:]

    y0 = np.array([1.1, 0, 0, 0])
    t0 = 0.0
    tfinal = 100.0  # Specify the final time
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
    return mod, sim, stability_index, [t, y]

if __name__ == '__main__':
    # run_elastic_pendulum_problem()
    # run_beam_problem_HHT('HHT', alpha=-1/3, with_plots=True, h=1)
    # mpl.show()

    # run_beam_problem_HHT('Newmark_implicit', 0, 0.25, 0.7,  with_plots=True)
    # mpl.show()

    if False:
        # generate a bunch of images to be included in the report
        # show damping in dependence of the alpha parameter
        _, _, soln = run_beam_problem_HHT('HHT', 0, h=1)
        plot_energies(soln[0], soln[3], soln[4], savefig=True, plotnumber=900)
        _, _, soln = run_beam_problem_HHT('HHT', -1/3, h=1)
        plot_energies(soln[0], soln[3], soln[4], savefig=True, plotnumber=901)
        # exact energy plot
        _, _, soln = run_beam_problem_HHT()
        plot_energies(soln[0], soln[3], soln[4], savefig=True, plotnumber=902)
        # exact displacement plot
        _, _, soln = run_beam_problem_HHT()
        plot_displacement(soln[0], soln[2], savefig=True, plotnumber=905)

    if False:
        # Test the alpha parameter on the HHT solver
        stability_indxs = []
        alphas = np.linspace(-1/3,0,10)
        for alpha in alphas:
            mod, sim, soln = run_beam_problem_HHT('HHT', alpha)
            stability_indxs.append(soln[6])
        mpl.figure()
        mpl.plot(alphas, stability_indxs)
        mpl.xlabel(r'$\alpha$')
        mpl.ylabel('Variance($E_{tot}$)')
        mpl.savefig(f'../Plots/Project3_main/Figure_920.pdf')
    
    if False:
        # Test the beta and gamma parameter on the implicit Newmark method
        # experiments are of the form [alpha, betas, gammas]
        experiments = [[0., np.linspace(0.01,0.49,20), np.linspace(0.5,0.99,20)]]
                        #   [0., np.linspace(0.01,0.49,8), np.linspace(0.7,0.99,8)]]
        
        for i, experiment in enumerate(experiments):
            stability_indxs = []
            alpha = experiment[0]
            betas = experiment[1]
            gammas = experiment[2]

            for j, beta in enumerate(betas):
                stability_indxs.append([])
                for gamma in gammas:
                    mod, sim, soln = run_beam_problem_HHT('Newmark_implicit', alpha, beta, gamma)
                    stability_indxs[j].append(soln[6])

            max_stab = 1E1
            stability_indxs = np.nan_to_num(np.asarray(stability_indxs))
            stability_indxs[stability_indxs >= max_stab] = max_stab
            stability_indxs[stability_indxs <= 0] = max_stab

            mpl.figure(i)
            fig, ax = mpl.subplots()
            X, Y = np.meshgrid(betas, gammas)
            surf = mpl.contourf(X, Y, np.log10(stability_indxs))
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(r'$\gamma$')
            fig.colorbar(surf, label=r'$log_{10}$(Variance($E_{tot}$))')
            mpl.savefig(f'../Plots/Project3_main/Figure_{910+i}.pdf')
    
    if True:
        # compare the different solver methods
        solvers = ['solver', 'HHT' , 'ImplicitEuler']
        stability_indxs = [r'Variance($E_{\text{tot}}$)']
        solving_times = ['Elapsed simulation time [s]']
        for solver in solvers[1:]:
            mod, sim, soln = run_beam_problem_HHT(solver)
            stability_indxs.append(f'{soln[6]:0.1e}')
            solving_times.append(np.around(soln[7], 1))
            
        # print(tabulate([solvers, stability_indxs, solving_times], headers='firstrow', tablefmt='fancy_grid'))
        with open('../Plots/Tables/Statistics_beam_solvers.tex', 'w') as output:
            output.write(tabulate([solvers, stability_indxs, solving_times], headers='firstrow', tablefmt='latex_raw'))




    if True:

        # This plots compare the explicit version of Newmarks Method with classical methods
        all_solns = []
        methods = []

        _, _, _, soln = run_elastic_pendulum_problem_newmark(k=10, with_plots=False)
        all_solns.append(soln)
        _, _, _, soln = run_elastic_pendulum_problem(solver=ExplicitEuler, k=10, with_plots=False)
        all_solns.append(soln)
        _, _, _, soln = run_elastic_pendulum_problem(solver=RungeKutta4, k=10, with_plots=False)
        all_solns.append(soln)

        t = np.linspace(0, 0.03, 500)
        all_solns_interp = np.zeros((3, t.size, 2))
        for i, soln in enumerate(all_solns):
            for j in range(2):
                all_solns_interp[i, :, j] = np.interp(t, soln[0], soln[1][:, j])

        # Plot soln
        mpl.plot(all_solns[0][0], all_solns[0][1])
        mpl.show()
        mpl.plot(all_solns[1][0], all_solns[1][1])
        mpl.show()
        mpl.plot(all_solns[2][0], all_solns[2][1])
        mpl.show()
        #plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[0, :, :], savefig=True, plotnumber=520)
        #plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[1, :, :], savefig=True, plotnumber=530)
        #plot_soln(t, all_solns_interp[3, :, :] - all_solns_interp[2, :, :], savefig=True, plotnumber=540)
        # mpl.show()
