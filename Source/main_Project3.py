import numpy as np
from assimulo.problem import Explicit_Problem
from Explicit_Problem_2nd import Explicit_Problem_2nd
from HHT import HHT, Newmark_explicit, Newmark_implicit
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl
import math
from main_Project1 import f_pendulum, plot_pendulum, pendulum_energies
from elastodyn import *
from tabulate import tabulate


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
    # run_elastic_pendulum_problem()
    # run_beam_problem_HHT('HHT', alpha=-1/3, with_plots=True, h=1)
    # mpl.show()

    run_beam_problem_HHT('HHT', with_plots=True)
    mpl.show()

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
            # plot_displacement(soln[0],soln[2])
            # mpl.show()
            # [tt, y, disp_tip, elastic_energy, kinetic_energy, total_energy, stability_index]
        # print(stability_indxs)
        mpl.figure()
        mpl.plot(alphas, stability_indxs)
        mpl.xlabel(r'$\alpha$')
        mpl.ylabel('stability_index')
        mpl.savefig(f'../Plots/Project3_main/Figure_920.pdf')
        mpl.show()
    
    if False:
        # Test the beta and gamma parameter on the implicit Newmark method
        stability_indxs = []
        alpha = 0
        n = 6
        m = n
        betas = np.linspace(0.1,0.49,n)
        gammas = np.linspace(0.2,0.99,m)
        for i, beta in enumerate(betas):
            stability_indxs.append([])
            for gamma in gammas:
                mod, sim, soln = run_beam_problem_HHT('Newmark_implicit', alpha, beta, gamma)
                stability_indxs[i].append(soln[6])
            # [tt, y, disp_tip, elastic_energy, kinetic_energy, total_energy, stability_index]
        max_stab = 1E2
        stability_indxs = np.nan_to_num(np.asarray(stability_indxs))
        stability_indxs[stability_indxs >= max_stab] = max_stab
        stability_indxs[stability_indxs <= 0] = max_stab
        # fig = mpl.figure()
        fig, ax = mpl.subplots()
        # fig, ax = mpl.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(betas, gammas)
        # Plot the surface.
        # color_matr = stability_indxs
        # color_matr[stability_indxs <= 0] = 1E2
        # colors = mpl.cm.jet(stability_indxs/float(stability_indxs.max()))
        # # colors[stability_indxs <0,:] = np.array([0., 1., 1., 1.])
        # surf = ax.plot_surface(X, Y, stability_indxs, facecolors=colors)
        surf = mpl.contourf(X, Y, stability_indxs)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\gamma$')
        # mpl.xlabel(r'$\beta$')
        # mpl.ylabel(r'$\gamma$')
        # ax.set_zlabel('stability_index')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='stability_index')
        mpl.savefig(f'../Plots/Project3_main/Figure_910.pdf')
        mpl.show()
    
    if False:
        # compare the different solver methods
        solvers = ['solver', 'HHT' , 'ImplicitEuler', 'Radau5ODE']
        stability_indxs = ['stability_index']
        solving_times = ['Elapsed simulation time [s]']
        for solver in solvers[1:]:
            mod, sim, soln = run_beam_problem_HHT(solver)
            stability_indxs.append(soln[6])
            solving_times.append(soln[7])
            
        # print(tabulate([solvers, stability_indxs, solving_times], headers='firstrow', tablefmt='fancy_grid'))
        with open('../Plots/Tables/Statistics_beam_solvers.tex', 'w') as output:
            output.write(tabulate([solvers, stability_indxs, solving_times], headers='firstrow', tablefmt='latex'))




    if False:
        # TODO: change this to compare solns of elastodyn

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