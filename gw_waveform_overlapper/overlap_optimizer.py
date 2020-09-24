import copy
from typing import List

import colorit
import numpy as np
import scipy.optimize

from . import overlap_computer
from .waveform import Waveform

TLIM = (-0.5, 0.5)
PLIM = (-np.pi, np.pi)


def fast_overlap_optimize(wf1: Waveform, wf2: Waveform, verbose=False,
                          method='Nelder-Mead'):
    """Method to estimate the maximum overlap.

    :param wf1:
    :param wf2:
    :return:
    """

    x0 = np.array([0, 0])
    path = [x0]

    if method == "Nelder-Mead":
        minimizer_kwargs = dict(
            args=(wf1, wf2),
            tol=1e-100,
            options=dict(disp=verbose, adaptive=True, maxiter=500),
            callback=make_minimize_cb(path),
            method='Nelder-Mead'
        )
    else:
        minimizer_kwargs = dict(
            args=(wf1, wf2),
            tol=1e-100,
            options=dict(disp=verbose, adaptive=True),
            callback=make_minimize_cb(path),
            bounds=[TLIM, PLIM],
            method='L-BFGS-B'
        )
    basin_bounds = BasinBounds()
    res = scipy.optimize.basinhopping(
        func=calculate_overlaps_optimizable,
        x0=path[-1],
        minimizer_kwargs=minimizer_kwargs,
        callback=print_fun,
        niter=10,
        stepsize=0.02,
        interval=3,
        accept_test=basin_bounds,
    )
    time_shift, phase_shift = res.x[0], res.x[1]
    maximum_overlap = -res.fun
    return time_shift, phase_shift, maximum_overlap, path


class BasinBounds(object):
    def __init__(self, xmin=[TLIM[0], PLIM[0]], xmax=[TLIM[1], PLIM[1]]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def make_minimize_cb(path=[]):
    def minimize_cb(xk):
        path.append(np.copy(xk))

    return minimize_cb


def print_fun(x, f, accepted):
    text = f"f({x[0]:.2f}, {x[1]:.2f}) = {f:0.2f}"
    if accepted:
        print("✔ " + colorit.color_front(text, 0, 255, 0))
    else:
        print("✘ " + colorit.color_front(text, 255, 0, 0))


def calculate_overlaps_optimizable(x: List, *args) -> float:
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    Optimisable function for calculating overlaps.

    x: List of the [timeshift, phaseshift]
    args: Tuple

    """
    wf1, wf2 = (args)
    temp_wf2 = copy.deepcopy(wf2)
    temp_wf2.time_shift(amount=x[0])
    temp_wf2.phase_shift(amount=x[1])
    return - overlap_computer.compute_overlap(wf1, temp_wf2)
