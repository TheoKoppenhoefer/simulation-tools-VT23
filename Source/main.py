import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as mpl


def run_elastic_pendulum_problem(with_plots=True, k=1, atol=1E-6, rtol=1E-6, maxord=5, discr='BDF'):
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
    sim.discr = discr
    sim.maxord = maxord
    sim.atol = [atol]
    sim.rtol = rtol

    # Simulate
    t, y = sim.simulate(tfinal)

    # calculate the energy
    pot_energy = 1+y[:,1]
    kin_energy = np.linalg.norm(y[:,2:4], axis=1)**2/2.
    elast_energy = k*(1-np.linalg.norm(y[:,0:2], axis=1))**2 /2.
    total_energy = pot_energy+kin_energy+elast_energy
    stability_index = (np.max(total_energy)-np.min(total_energy))/np.mean(total_energy)

    # Plot
    if with_plots:
        for i in range(y.shape[1]):
            mpl.plot(t, y[:, i])
        mpl.legend([r'$y_1$', r'$y_2$', r'$\dot{y}_1$', r'$\dot{y}_2$'])
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
    return mod, sim, stability_index

def plot_stats(xdata, ydata, xlabel='x', plotlabel='', plotnumber=100, semilogx=False, savefig=False):
    ylabels = ['nsteps', 'nfcns', 'njacs', 'nerrfails', 'instability index']
    for i in range(5):
        mpl.figure(plotnumber+i, clear=False)
        if semilogx:
            mpl.semilogx(xdata, ydata[i], label=plotlabel)
        else:
            mpl.plot(xdata, ydata[i], label=plotlabel)
        mpl.legend()
        mpl.ylabel(ylabels[i])
        mpl.xlabel(xlabel)
        if savefig:
            mpl.savefig(f'../Plots/Figure_{plotnumber+i}.pdf')
    ylabels = ['nfcns / nsteps', 'njacs / nsteps', 'nerrfails / nsteps']
    for i in range(3):
        mpl.figure(plotnumber+10+i, clear=False)
        if semilogx:
            mpl.semilogx(xdata, np.asarray(ydata[i+1]) / np.asarray(ydata[0]), label=plotlabel)
        else:
            mpl.plot(xdata, np.asarray(ydata[i + 1]) / np.asarray(ydata[0]), label=plotlabel)
        mpl.legend()
        mpl.ylabel(ylabels[i])
        mpl.xlabel(xlabel)
        if savefig:
            mpl.savefig(f'../Plots/Figure_{plotnumber+10+i}.pdf')


if __name__ == '__main__':
    run_elastic_pendulum_problem()

    maxords = {'BDF': [3,4,5], 'Adams': [3,6,12]}
    for discr in ['BDF', 'Adams']:
        for maxord in maxords[discr]:
            ks = []
            nsteps = []
            nfcns = []
            njacs = []
            nerrfails = []
            stability_indexs = []
            for k in range(1, int(1E3), int(1E2)):
                # Test ATOL
                ks.append(k)
                mod, sim, stability_index = run_elastic_pendulum_problem(discr=discr, maxord=maxord, k=k, with_plots=False)
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


        rtols = np.logspace(1E-8,1,10)
        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        stability_indexs = []
        for rtol in rtols:
            # Test RTOL
            mod, sim, stability_index = run_elastic_pendulum_problem(k=1E2, discr=discr, rtol=rtol, with_plots=False)
            stats = sim.get_statistics()
            nsteps.append(stats.__getitem__('nsteps'))
            nfcns.append(stats.__getitem__('nfcns'))
            njacs.append(stats.__getitem__('njacs'))
            nerrfails.append(stats.__getitem__('nerrfails'))
            stability_indexs.append(stability_index)

        # Plot the whole lot
        plot_stats(rtols, [nsteps, nfcns, njacs, nerrfails, stability_indexs],
                   xlabel='rtol', plotlabel=f'rtols, discr={discr}',plotnumber=300, semilogx=True,
                   savefig=True)


        atols = np.logspace(1E-8,1,10)
        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        stability_indexs = []
        for atol in atols:
            # Test ATOL
            mod, sim, stability_index = run_elastic_pendulum_problem(k=1E2, discr=discr, atol=atol, with_plots=False)
            stats = sim.get_statistics()
            nsteps.append(stats.__getitem__('nsteps'))
            nfcns.append(stats.__getitem__('nfcns'))
            njacs.append(stats.__getitem__('njacs'))
            nerrfails.append(stats.__getitem__('nerrfails'))
            stability_indexs.append(stability_index)

        # Plot the whole lot
        plot_stats(atols, [nsteps, nfcns, njacs, nerrfails, stability_indexs],
                   xlabel='atol', plotlabel=f'atol, discr={discr}',plotnumber=400, semilogx=True,
                   savefig=True)
    #mpl.show()