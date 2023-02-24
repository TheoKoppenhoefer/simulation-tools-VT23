from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import fsolve
import numpy.linalg as nl
import math
from scipy.sparse.linalg import spsolve

class Newmark_implicit(Explicit_ODE):
    """
    Base class for the HHT-alpha and the Newmark method solvers
    """
    tol = 1.e-4
    maxit = 100
    maxsteps = 50000


    def __init__(self, problem):
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

    def _set_alpha(self, alpha):
        self.options["alpha"] = float(alpha)

    def _get_alpha(self):
        return self.options["alpha"]

    def _set_beta(self, beta):
        self.options["beta"] = float(beta)

    def _get_beta(self):
        return self.options["beta"]

    def _set_gamma(self, gamma):
        self.options["gamma"] = float(gamma)

    def _get_gamma(self):
        return self.options["gamma"]

    h = property(_get_h, _set_h)
    alpha = property(_get_alpha, _set_alpha)
    beta = property(_get_beta, _set_beta)
    gamma = property(_get_gamma, _set_gamma)

    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """

        u = self.problem.u0
        ud = self.problem.ud0
        udd = self.get_udd0(t, u, ud)

        # Lists for storing the result
        tres = []
        yres = []

        h = self.options["h"]

        for i in range(self.maxsteps):
            tres.append(t)
            yres.append(np.concatenate((u,ud)))

            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            h = min(h, np.abs(tf - t))

            u, ud, udd= self.step(t, u, ud, udd)
            t += h

        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres

    def step(self, t, u_nm1, ud_nm1, udd_nm1):
        
        u_n = self.get_u_n(t, u_nm1, ud_nm1, udd_nm1)
        ud_n = self.get_ud_n(u_n, u_nm1, ud_nm1, udd_nm1)
        udd_n = self.get_udd_n(u_n, u_nm1, ud_nm1, udd_nm1)
        
        return u_n, ud_n, udd_n

    def get_udd0(self, t, u_0, ud_0):
        h = self.options["h"]
        alpha = self.options["alpha"]
        beta = self.options["beta"]
        gamma = self.options["gamma"]
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        f = self.problem.f

        # Update statistics
        self.statistics["nfcns"] += 1

        return spsolve(M, f(t)-C@ud_0-K@u_0)

    def get_u_n(self, t, u_nm1, ud_nm1, udd_nm1):
        h = self.options["h"]
        alpha = self.options["alpha"]
        beta = self.options["beta"]
        gamma = self.options["gamma"]
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        f = self.problem.f
        lhs_Matrix = M/(beta*h**2)+gamma/(beta*h)*C+(1+alpha)*K

        # Update statistics
        self.statistics["nfcns"] += 1

        # Solve (8'') to get u_np1
        return spsolve(lhs_Matrix, f(t) \
                        + M@( u_nm1/(beta*h**2+ud_nm1/(beta*h)+(1/(2*beta)-1)*udd_nm1) ) \
                        + C@( u_nm1* gamma/(beta*h) -(1-gamma/beta)*ud_nm1-(1-gamma/(2*beta))*h*udd_nm1 )
                        + alpha*K@ u_nm1)

    def get_ud_n(self, u_n, u_nm1, ud_nm1, udd_nm1):
        h = self.options["h"]
        alpha = self.options["alpha"]
        beta = self.options["beta"]
        gamma = self.options["gamma"]
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        f = self.problem.f

        # solve (7') to get ud_np1
        return gamma/beta*(u_n-u_nm1)/h+(1-gamma/beta)*ud_nm1+(1-gamma/(2*beta))*h*udd_nm1

    def get_udd_n(self, u_n, u_nm1, ud_nm1, udd_nm1):
        h = self.options["h"]
        alpha = self.options["alpha"]
        beta = self.options["beta"]
        gamma = self.options["gamma"]
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        f = self.problem.f

        # solve (6') to get udd_np1
        return (u_n-u_nm1)/(beta*h**2)-ud_nm1/(beta*h)-(1/(2*beta)-1)*udd_nm1



    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF2', verbose)
        self.log_message(' Solver type       : Fixed step\n', verbose)

class Newmark_explicit(Newmark_implicit):
    alpha = None
    
    def step(self, t, u_nm1, ud_nm1, udd_nm1):
        
        u_n = self.get_u_n(t, u_nm1, ud_nm1, udd_nm1)
        udd_n = self.get_udd_n(t, u_n)
        ud_n = self.get_ud_n(ud_nm1, udd_n, udd_nm1)
        
        return u_n, ud_n, udd_n

    def get_u_n(self, t, u_nm1, ud_nm1, udd_nm1):
        h = self.options["h"]
        return u_nm1+ud_nm1*h+udd_nm1*h**2/2

    def get_ud_n(self, ud_nm1, udd_n, udd_nm1):
        h = self.options["h"]
        return ud_nm1+udd_nm1*h/2+udd_n*h/2

    def get_udd_n(self, t, u_n):
        h = self.options["h"]
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        f = self.problem.f

        # Update statistics
        self.statistics["nfcns"] += 1

        # Solve (8'') to get u_np1
        return spsolve(M, f(t)-K@u_n)

class HHT(Newmark_implicit):
    beta = None
    gamma = None
    
    def _set_alpha(self, alpha):
        self.options["alpha"] = alpha
        self.options["beta"] = ((1-alpha)/2)**2
        self.options["gamma"] = 1/2-alpha

    alpha = property(Newmark_implicit._get_alpha, _set_alpha)
