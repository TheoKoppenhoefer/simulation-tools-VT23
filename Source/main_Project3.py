import numpy as np
from assimulo.problem import Explicit_Problem
from Explicit_Problem_2nd import Explicit_Problem_2nd
from HHT import HHT, Newmark_explicit, Newmark_implicit
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl
import math
from main_Problem1 import rhs


def run_elastic_pendulum_problem(with_plots=True, k=1., atol=1E-6, rtol=1E-6, maxord=5, discr='BDF'):
    """
    """
    def get_udd_n(u_n, ud_n):
        return rhs(0, np.concatenate((u_n,ud_n)))[2:]

    def get_udd0(u0, ud0):
        return rhs(0, np.concatenate((u0,ud0)))[2:]

    y0 = np.array([1.1, 0, 0, 0])
    t0 = 0.0
    tfinal = 10.0  # Specify the final time
    # Define an Assimulo problem
    u0 = np.array([1.1, 0])
    ud0 = np.zeros(2)

    mod = Explicit_Problem_2nd(None, None, None, u0, ud0, t0, None, name=r'Elastic Pendulum Problem')

    # Define an explicit solver
    sim = Newmark_explicit(mod)

    # Sets the parameters
    sim.get_udd0 = get_udd0
    sim.get_udd_n = get_udd_n

    # Simulate
    t, y = sim.simulate(tfinal)

    # calculate the energy
    pot_energy = 1 + y[:, 1]
    kin_energy = np.linalg.norm(y[:, 2:4], axis=1) ** 2 / 2.
    elast_energy = k * (1 - np.linalg.norm(y[:, 0:2], axis=1)) ** 2 / 2.
    total_energy = pot_energy + kin_energy + elast_energy
    stability_index = (np.max(total_energy) - np.min(total_energy)) / np.mean(total_energy)

    # Plot
    if with_plots:
        for i in range(y.shape[1]):
            mpl.plot(t, y[:, i])
        mpl.legend([r'$y_1$', r'$y_2$', r'$\dot{y}_1$', r'$\dot{y}_2$'])
        mpl.title('Cartesian Coordinates')
        mpl.show()
        mpl.plot(y[:, 0], y[:, 1])
        mpl.xlabel(r'$y_1$')
        mpl.ylabel(r'$y_2$')
        # mpl.title('phase portrait')
        mpl.show()
        # Energy plot
        mpl.plot(t, pot_energy, label='potential energy')
        mpl.plot(t, kin_energy, label='kinetic energy')
        mpl.plot(t, elast_energy, label='elastic energy')
        mpl.plot(t, total_energy, label='total energy')
        mpl.xlabel(r'$t$')
        mpl.ylabel(r'Energy')
        mpl.legend()
        mpl.show()
        # Polar Coordiantes
        mpl.plot(t, np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2))
        mpl.plot(t, np.arctan2(y[:, 1], y[:, 0]) + math.pi / 2)
        mpl.xlabel(r'$t$')
        mpl.legend([r'$r$', r'$theta$'])
        mpl.title('Polar Coordinates')
        mpl.show()
    return mod, sim, stability_index


def plot_stats(xdata, ydata, xlabel='x', plotlabel='', plotnumber=100, semilogx=False, savefig=False):
    ylabels = ['nsteps', 'nfcns', 'njacs', 'nerrfails', 'instability index']
    for i in range(5):
        mpl.figure(plotnumber + i, clear=False)
        if semilogx:
            mpl.semilogx(xdata, ydata[i], label=plotlabel)
        else:
            mpl.plot(xdata, ydata[i], label=plotlabel)
        mpl.legend()
        mpl.ylabel(ylabels[i])
        mpl.xlabel(xlabel)
        if savefig:
            mpl.savefig(f'../Plots/Task4/Figure_{plotnumber + i}.pdf')
    ylabels = ['nfcns / nsteps', 'njacs / nsteps', 'nerrfails / nsteps']
    for i in range(3):
        mpl.figure(plotnumber + 10 + i, clear=False)
        if semilogx:
            mpl.semilogx(xdata, np.asarray(ydata[i + 1]) / np.asarray(ydata[0]), label=plotlabel)
        else:
            mpl.plot(xdata, np.asarray(ydata[i + 1]) / np.asarray(ydata[0]), label=plotlabel)
        mpl.legend()
        mpl.ylabel(ylabels[i])
        mpl.xlabel(xlabel)
        if savefig:
            mpl.savefig(f'../Plots/Task4/Figure_{plotnumber + 10 + i}.pdf')


if __name__ == '__main__':
    run_elastic_pendulum_problem()

    maxords = {'BDF': [3, 4, 5], 'Adams': [3, 6, 12]}
    for discr in ['BDF', 'Adams']:
        if False:
            for maxord in maxords[discr]:
                ks = []
                nsteps = []
                nfcns = []
                njacs = []
                nerrfails = []
                stability_indexs = []
                for k in np.linspace(1,2E3,20):
                    # Test ATOL
                    ks.append(k)
                    mod, sim, stability_index = run_elastic_pendulum_problem(discr=discr, maxord=maxord, k=k,
                                                                             with_plots=False)
                    stats = sim.get_statistics()
                    nsteps.append(stats.__getitem__('nsteps'))
                    nfcns.append(stats.__getitem__('nfcns'))
                    njacs.append(stats.__getitem__('njacs'))
                    nerrfails.append(stats.__getitem__('nerrfails'))
                    stability_indexs.append(stability_index)

                # Plot the whole lot
                plot_stats(ks, [nsteps, nfcns, njacs, nerrfails, stability_indexs],
                           xlabel=r'$k$', plotlabel=f'discr={discr}, maxord={maxord}', plotnumber=200,
                           savefig=True)

        if False:
            rtols = np.logspace(1E-10, 1, 30)
            nsteps = []
            nfcns = []
            njacs = []
            nerrfails = []
            stability_indexs = []
            for rtol in rtols:
                # Test RTOL
                mod, sim, stability_index = run_elastic_pendulum_problem(k=1E3, discr=discr, rtol=rtol, with_plots=False)
                stats = sim.get_statistics()
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
                stability_indexs.append(stability_index)

            # Plot the whole lot
            plot_stats(rtols, [nsteps, nfcns, njacs, nerrfails, stability_indexs],
                       xlabel='rtol', plotlabel=f'discr={discr}', plotnumber=300, semilogx=True,
                       savefig=True)

        if False:
            atols = np.logspace(1E-8, 0.5, 30)
            nsteps = []
            nfcns = []
            njacs = []
            nerrfails = []
            stability_indexs = []
            for atol in atols:
                # Test ATOL
                mod, sim, stability_index = run_elastic_pendulum_problem(k=1E3, discr=discr, atol=atol, with_plots=False)
                stats = sim.get_statistics()
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
                stability_indexs.append(stability_index)

            # Plot the whole lot
            plot_stats(atols, [nsteps, nfcns, njacs, nerrfails, stability_indexs],
                       xlabel='atol', plotlabel=f'discr={discr}', plotnumber=400, semilogx=True,
                       savefig=True)
    if False:
        run_elastic_pendulum_problem(k=1E3, atol=1E-2)
        run_elastic_pendulum_problem(k=1E3)
    # mpl.show()
