from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import fsolve
import numpy.linalg as nl
import math

class 2nd_Order(Explicit_ODE):
    """
    Base class for the HHT-alpha and the Newmark method solvers
    """
    tol = 1.e-4
    maxit = 100
    maxsteps = 50000

    def __init__(self, problem, h=0.01, alpha=0, beta=0.5, gamma=0.5):
        Explicit_ODE.__init__(self, problem)  # Calls the base class

        # Solver options
        self.options["h"] = 0.01
        self.options["alpha"] = 0.
        self.options["beta"] = 0.5
        self.options["gamma"] = 0.5

        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

    def _set_alpha(self, alpha):
        self.options["alpha"] = float(alpha)

    def _get_alpha(self):
        return self.options["alpha"]

    alpha = property(_get_alpha, _set_alpha)

    def _set_beta(self, beta):
        self.options["beta"] = float(beta)

    def _get_beta(self):
        return self.options["beta"]

    beta = property(_get_beta, _set_beta)

    def _set_gamma(self, gamma):
        self.options["gamma"] = float(gamma)

    def _get_gamma(self):
        return self.options["gamma"]

    gamma = property(_get_gamma, _set_gamma)

    def integrate(self, t_np1, y_np1, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]

        # Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            tres.append(t_np1)
            yres.append(y_np1)

            if t_np1 >= tf:
                break
            self.statistics["nsteps"] += 1
            h = min(h, np.abs(tf - t_np1))
            k = min(i + 1, order)
            # shift
            T = np.concatenate(([t_np1], T[:k - 1]))
            Y = np.concatenate(([y_np1], Y[:k - 1, :]))

            t_np1, y_np1 = self.step_BDFk(T, Y, h, alpha[k - 1])

        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres

    def step_BDFk(self, T, Y, h, alpha):
        pass

    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF2', verbose)
        self.log_message(' Solver type       : Fixed step\n', verbose)

