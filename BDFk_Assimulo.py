from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import fsolve
import numpy.linalg as nl

class BDF_k(Explicit_ODE):
    """
    BDF-k   (Example of how to set-up own integrators for Assimulo)
    """
    tol = 1.e-8
    maxit = 100
    maxsteps = 500

    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem)  # Calls the base class

        # Solver options
        self.options["h"] = 0.01
        self.options["order"] = 2

        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

    def _set_h(self, h):
        self.options["h"] = float(h)

    def _set_order(self, order):
        self.options["order"] = int(order)

    def _get_h(self):
        return self.options["h"]

    def _get_order(self):
        return self.options["order"]

    h = property(_get_h, _set_h)

    def integrate(self, t_np1, y_np1, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        order = self.options["order"]

        # Lists for storing the result
        tres = []
        yres = []
        T = []
        Y = np.expand_dims(y_np1, 0)
        alpha = [[-1, 1], [3 / 2, -2, 1 / 2], [11 / 6, -3, 3 / 2, -1 / 3], [25 / 12, -4, 3, -4 / 3, 1 / 4],
                 [137 / 60, -5, 5, -10 / 3, 5 / 4, -1 / 5], [49 / 20, -6, 15 / 2, -20 / 3, 15 / 4, -6 / 5, 1 / 6]]

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
        """
        BDF-k with Newton's method (to be implemented) and Zero order predictor
        here are some values for alpha
        k=1: alpha=[-1,1]
        k=2: alpha=[3/2,-2,1/2]
        k=3: alpha=[11/6,-3,3/2,-1/3]
        k=4: alpha=[25/12,-4,3,-4/3,1/4]
        k=5: alpha=[137/60,-5,5,-10/3,5/4,-1/5]
        k=6: alpha=[49/20,-6,15/2,-20/3,15/4,-6/5,1/6]
        source: http://www.scholarpedia.org/article/Backward_differentiation_formulas
        """
        f = self.problem.rhs
        # predictor
        t_np1 = T[0] + h
        y_np1 = Y[0]  # zero order predictor
        # corrector with fixed point iteration
        alpha_cdot_Y = np.dot(alpha[1:], Y)
        # the following should be replaced by newton iteration
        y_np1 = fsolve(lambda x: h * f(t_np1, x) - alpha_cdot_Y - alpha[0] * x
                       , y_np1, xtol=self.tol)
        # y_np1 = classical_newton_optimisation(lambda x: h * f(t_np1, x) - alpha_cdot_Y - alpha[0] * x
        #               , y_np1, residual_eps=self.tol)

        return t_np1, y_np1

    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name), verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : ' + str(self.statistics["nsteps"]), verbose)
        self.log_message(' Number of Function Evaluations : ' + str(self.statistics["nfcns"]), verbose)

        self.log_message('\nSolver options:\n', verbose)
        self.log_message(' Solver            : BDF2', verbose)
        self.log_message(' Solver type       : Fixed step\n', verbose)

def finite_difference_gradient(func, x_arg, epsilon=1e-12):
    """
    Implementation of finite difference because scipy.optimisation.approx_fprime is useless
    :param func: (scalar) Function to differentiate, has to be vectorised in such a way that it can handle input of
                 the form [[x1,y1,...],...,[xd,yd,...]]
    :param x_arg: the coordinates [x1, ..., xd] or coordinate list [[x1,y1,...],...,[xd,yd,...]] at which to differentiate
    :param epsilon: the finite difference step size
    :return: A gradient of the form [f1,...,fd] or [[f1,g1,...],...,[fd,gd,...]]
    """
    # TODO: Implement the following using symmetric gradients
    x = np.asarray(x_arg)
    vectorized = True
    if x.ndim == 0:
        x = np.expand_dims(x, axis=0)
    if x.ndim == 1:
        vectorized = False
        x = np.expand_dims(x, axis=1)
    dim = x.shape[0]
    left = np.tile(np.expand_dims(x, axis=1), (dim, 1))
    right = np.expand_dims(np.eye(dim), list(range(2, left.ndim)))
    temp_arr = left + epsilon * right
    soln = (func(temp_arr) - func(left)) / epsilon
    # if not vectorized:
    #     return np.squeeze(soln, axis=-1)
    return soln


def finite_difference_hessian(gradient, x_arg, epsilon=1e-6):
    """
    Implementation of finite difference to calculate the hessian
    :param gradient: a gradient function that handles input in the same way as finite_difference_gradient does
    :param x_arg: coordinates of the form [x1,...,xd] at which to calculate the hessian
    :param epsilon: the finite difference step size. This has to be smaller than the step size of the
    finite_difference_gradient function, if used
    :return: A dxd numpy jacobian matrix
    """
    # TODO: Implement the following as symmetric gradients
    x = np.asarray(x_arg)
    x = np.expand_dims(x, axis=1)
    dim = x.shape[0]
    left = np.tile(np.expand_dims(x, axis=1), (dim, 1))
    right = np.expand_dims(np.eye(dim), list(range(2, left.ndim)))
    temp_arr = left + epsilon * right
    soln = (gradient(np.squeeze(temp_arr)) - gradient(x)) / epsilon
    return (soln + soln.transpose()) / 2.


def classical_newton_optimisation(objective, x0, cauchy_eps=1E-6, residual_eps=1E-6, maxiter=100):
    """

    :param objective:
    :param x0:
    :param tol:
    :param maxiter:
    :return:
    """
    gradient = lambda x: finite_difference_gradient(objective, x)
    hessian = lambda y: finite_difference_hessian(gradient, y)
    xk = x0
    for i in range(maxiter):
        gk = gradient(xk)
        sk = -nl.linalg.solve(hessian(xk), gk)
        xk += sk
        if nl.norm(sk) < cauchy_eps and nl.norm(gk) < residual_eps:
            break
    else:
        raise Exception('Newton iteration did not converge')

if __name__ == '__main__':
    # Define the rhs
    def f(t, y):
        ydot = -y[0]
        return np.array([ydot])


    # Define an Assimulo problem
    exp_mod = Explicit_Problem(f, 4)
    exp_mod.name = 'Simple BDF-k Example'


    # Define another Assimulo problem
    def pend(t, y):
        # g=9.81    l=0.7134354980239037
        gl = 13.7503671
        return np.array([y[1], -gl * np.sin(y[0])])


    pend_mod = Explicit_Problem(pend, y0=np.array([2. * np.pi, 1.]))
    pend_mod.name = 'Nonlinear Pendulum'

    # Define an explicit solver
    exp_sim = BDF_k(pend_mod)  # Create a BDF solver
    t, y = exp_sim.simulate(1)
    exp_sim.plot()
    mpl.show()
