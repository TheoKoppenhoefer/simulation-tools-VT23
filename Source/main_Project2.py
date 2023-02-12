import numpy as np
from assimulo.solvers import IDA, RungeKutta4, CVode
from squeezer import Seven_bar_mechanism
from squeezer2 import Seven_bar_mechanism_indx2
from squeezer1 import Seven_bar_mechanism_indx1, Seven_bar_mechanism_expl
import matplotlib.pyplot as mpl

var_labels = [r'$\beta$', r'$\Theta$', r'$\gamma$', r'$\phi$', r'$\delta$', r'$\Omega$', r'$\epsilon$',
              r'$\dot{\beta}$', r'$\dot{\Theta}$', r'$\dot{\phi}$', r'$\dot{\delta}$', r'$\dot{\omega}$',
              r'$\dot{\Omega}$', r'$\dot{\epsilon}$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
              r'$\lambda_4$', r'$\lambda_5$', r'$\lambda_6$', r'$\lambda_7$']


def run_seven_bar_problem(with_plots=True, problem_index=3, atol_v=1E-6, atol_lambda=1E-6,
                          algvar_v=False, algvar_lambda=False, suppress_alg=True):
    """
    """
    tfinal = 0.03  # Specify the final time

    if problem_index <= 0:
        mod = Seven_bar_mechanism_expl()
    elif problem_index == 1:
        mod = Seven_bar_mechanism_indx1()
    elif problem_index == 2:
        mod = Seven_bar_mechanism_indx2()
    else:
        mod = Seven_bar_mechanism()

    if problem_index <= 0:
        # Define an explicit solver
        sim = CVode(mod)

        # Set the parameters

        # Simulate
        t, y = sim.simulate(tfinal)
    else:
        # Define an implicit solver
        sim = IDA(mod)

        # Set the parameters
        atol = 1E-6 * np.ones((20,))
        atol[7:14] = atol_v
        atol[14:20] = atol_lambda
        sim.atol = atol
        algvar = np.ones((20,))
        algvar[7:14] = algvar_v
        algvar[14:20] = algvar_lambda
        sim.algvar = algvar
        sim.suppress_alg = suppress_alg

        # Simulate
        t, y, yd = sim.simulate(tfinal)

    # Plot
    if with_plots:
        # do some plotting
        for i in range(7):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[:7])
        mpl.figure()
        for i in range(7, 14):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[7:14])
        mpl.figure()
        if problem_index > 0:
            for i in range(14, 20):
                mpl.plot(t, y[:, i])
            mpl.legend(var_labels[14:])
        mpl.show()
    return mod, sim


def plot_stats(xdata, ydata, plotnumber=100, savefig=False):
    ylabels = ['nsteps', 'nfcns', 'njacs', 'nerrfails']
    for i in range(4):
        mpl.figure(plotnumber + i, clear=False)
        # fig, ax = mpl.subplots()
        mpl.bar(xdata, ydata[i])
        mpl.ylabel(ylabels[i])
        if savefig:
            mpl.savefig(f'../Plots/Project2_main/Figure_{plotnumber + i}.pdf')
    ylabels = ['nfcns / nsteps', 'njacs / nsteps', 'nerrfails / nsteps']
    for i in range(3):
        mpl.figure(plotnumber + 10 + i, clear=False)
        mpl.bar(xdata, np.asarray(ydata[i + 1]) / np.asarray(ydata[0]))
        mpl.ylabel(ylabels[i])
        if savefig:
            mpl.savefig(f'../Plots/Project2_main/Figure_{plotnumber + 10 + i}.pdf')


if __name__ == '__main__':
    run_seven_bar_problem(True, 2, 1E5, 1E5, False, False, True)

    if False:
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [1, 1E-6, 1E5, False, True, True],
                       [2, 1E-6, 1E5, False, False, True],
                       [2, 1E-6, 1E5, False, True, True],
                       [2, 1E-6, 1E5, False, True, True],
                       [3, 1E5, 1E5, False, False, True],
                       [3, 1E5, 1E-6, True, False, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for exp in experiments:
            try:
                mod, sim = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(
                    f'problem_index={exp[0]} \n  atol_v={exp[1]:.0E} \n atol_lambda={exp[2]:.0E} \n algvar_v={exp[3]} \n algvar_lambda={exp[4]} \n suppress_alg={exp[5]}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails])
        mpl.show()
