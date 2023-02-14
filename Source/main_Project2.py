import numpy as np
from assimulo.solvers import IDA, RungeKutta4, CVode
from squeezer import Seven_bar_mechanism_indx3, Seven_bar_mechanism_indx2, Seven_bar_mechanism_indx1, Seven_bar_mechanism_expl
import matplotlib.pyplot as mpl

var_labels = [r'$\beta$', r'$\Theta$', r'$\gamma$', r'$\phi$', r'$\delta$', r'$\Omega$', r'$\epsilon$',
              r'$\dot{\beta}$', r'$\dot{\Theta}$', r'$\dot{\phi}$', r'$\dot{\delta}$', r'$\dot{\omega}$',
              r'$\dot{\Omega}$', r'$\dot{\epsilon}$', r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
              r'$\lambda_4$', r'$\lambda_5$', r'$\lambda_6$', r'$\lambda_7$']


def run_seven_bar_problem(with_plots=True, problem_index=3, atol_v=1E5, atol_lambda=1E5,
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
        mod = Seven_bar_mechanism_indx3()

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
        plot_soln(t, y)
        mpl.show()
    return mod, sim, [t, y]

def plot_soln(t, y, savefig=False, plotnumber=500):
    # do some plotting
    for i in range(7):
        mpl.plot(t, y[:, i])
    mpl.legend(var_labels[:7])
    if savefig:
        mpl.savefig(f'../Plots/Project2_main/Figure_{plotnumber}.pdf')
    mpl.figure()
    for i in range(7, 14):
        mpl.plot(t, y[:, i])
    mpl.legend(var_labels[7:14])
    if savefig:
        mpl.savefig(f'../Plots/Project2_main/Figure_{plotnumber + 1}.pdf')
    if y.shape[1] > 14:
        mpl.figure()
        for i in range(14, 20):
            mpl.plot(t, y[:, i])
        mpl.legend(var_labels[14:])
        if savefig:
            mpl.savefig(f'../Plots/Project2_main/Figure_{plotnumber + 2}.pdf')
    


def plot_stats(xdata, ydata, plotnumber=500, savefig=False):
    ylabels = ['nsteps', 'nfcns', 'njacs', 'nerrfails']
    for i in range(4):
        mpl.figure(plotnumber + i, clear=False)
        # fig, ax = mpl.subplots(plotnumber + i, clear=False)
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
    # run_seven_bar_problem(True, 3, 1E5, 1E5, False, False, True)

    if True:
        # This plots comparisons of the index 1,2,3 formulations
        _, _, indx1_soln = run_seven_bar_problem(False, 1)
        _, _, indx2_soln = run_seven_bar_problem(False, 2)
        _, _, indx3_soln = run_seven_bar_problem(False, 3)
        
        t = np.linspace(0, 0.03, 500)
        all_solns = np.zeros((3,t.size, 20))
        for i, soln in enumerate([indx1_soln, indx2_soln, indx3_soln]):
            for j in range(20):
                all_solns[i, :, j] = np.interp(t, soln[0], soln[1][:, j])
        
        # Plot soln
        plot_soln(indx2_soln[0], indx2_soln[1], savefig=True, plotnumber=510)
        plot_soln(t, all_solns[1, :, :]-all_solns[0, :, :], savefig=True, plotnumber=520)
        plot_soln(t, all_solns[2, :, :]-all_solns[1, :, :], savefig=True, plotnumber=530)
        mpl.show()


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
                xdata.append(f'index {exp[0]}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=600, savefig=True)

    if False:
        # This tests the index=1 problem
        # list of experiments in the form [problem_index, atol_v, atol_lambda, algvar_v, algvar_lambda, suppress_alg]
        experiments = [[1, 1E5, 1E5, False, False, True],
                       [1, 1E-6, 1E5, False, True, True]]

        nsteps = []
        nfcns = []
        njacs = []
        nerrfails = []
        xdata = []
        for counter, exp in enumerate(experiments):
            try:
                mod, sim, _ = run_seven_bar_problem(False, *exp)

                stats = sim.get_statistics()
                xdata.append(f'problem {counter}')
                nsteps.append(stats.__getitem__('nsteps'))
                nfcns.append(stats.__getitem__('nfcns'))
                njacs.append(stats.__getitem__('njacs'))
                nerrfails.append(stats.__getitem__('nerrfails'))
            except:
                print(f'There seems to be a problem in the experiment {exp}')

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=700, savefig=True)

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

        plot_stats(xdata, [nsteps, nfcns, njacs, nerrfails], plotnumber=800, savefig=True)
    # mpl.show()


