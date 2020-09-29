import copy
from typing import List

import colorit
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

from . import overlap_computer
from .waveform import Waveform, create_similar_waveform

TLIM = (-4, 4)
PLIM = (-2 * np.pi, 2 * np.pi)
MAX_ITR = 500
TOL = 1e-100
NITER = 10
NUM = 25


def fft_overlap_optimizer(wf1: Waveform, wf2: Waveform, verbose=False):
    """ Gets Max overlap between two waveforms
    FINDCHIRP :https://arxiv.org/pdf/gr-qc/0509116.pdf

    z --> eq 4.2

    :param wf1:
    :param wf2:
    :param verbose:
    :return:
    """
    t0s = np.linspace(- wf1.duration / 2, wf1.duration / 2, num=100)
    wf1_temp = create_similar_waveform(wf1, dict(phase=0))
    wf2_temp = create_similar_waveform(wf2, dict(phase=0, geocent_time=0))
    z = np.vectorize(overlap_computer.complex_filter)
    zs = z(t0s, wf1_temp, wf2_temp)

    max_idx = np.argmax(np.abs(zs))
    time = t0s[max_idx]
    phase = np.angle(zs[max_idx])  # -pi, pi
    phase = np.mod(phase, 2 * np.pi)  # -, 2pi

    wf1_temp = create_similar_waveform(wf1_temp, dict(geocent_time=time, phase=phase))
    path = [[0, 0], [time, phase]]
    return time, phase, overlap_computer.compute_overlap(wf1_temp, wf2_temp), path


def overlap_optimizer(wf1: Waveform, wf2: Waveform, verbose=False,
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
            tol=TOL,
            options=dict(disp=verbose, adaptive=True, maxiter=MAX_ITR),
            callback=make_minimize_cb(path),
            method='Nelder-Mead'
        )
    else:
        minimizer_kwargs = dict(
            args=(wf1, wf2),
            tol=TOL,
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
        niter=NITER,
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
    temp_wf1 = copy.deepcopy(wf1)
    temp_wf1.time_shift(amount=x[0])
    temp_wf1.phase_shift(amount=x[1])
    return - overlap_computer.compute_overlap(temp_wf1, wf2)


def plot_waveform_optimization(wf1: Waveform, wf2: Waveform, path, fname):
    path = np.array(path).T

    t, p = np.meshgrid(
        np.linspace(*TLIM, num=NUM),
        np.linspace(*PLIM, num=NUM)
    )
    wf1 = create_similar_waveform(wf1, dict(phase=0))
    wf2 = create_similar_waveform(wf2, dict(phase=0, geocent_time=0))

    @np.vectorize
    def f(t_, p_):
        temp_wf1 = copy.deepcopy(wf1)
        temp_wf1.time_shift(t_)
        temp_wf1.phase_shift(p_)
        return overlap_computer.compute_overlap(temp_wf1, wf2)

    z = f(t, p)
    plt.contourf(t, p, z)

    fig, ax = plt.subplots(figsize=(10, 6))
    quadcontourset = ax.contourf(t, p, z)
    ax.quiver(path[0, :-1], path[1, :-1], path[0, 1:] - path[0, :-1],
              path[1, 1:] - path[1, :-1], scale_units='xy', angles='xy', scale=1,
              color='k')
    ax.plot(path[0, -1], path[-1, -1], 'r*', markersize=10)
    ax.plot(path[0, 0], path[-1, 0], 'b*', markersize=10)
    ax.set_xlabel('time')
    ax.set_ylabel('phase')
    ax.set_xlim(*TLIM)
    ax.set_ylim(*PLIM)
    clb = fig.colorbar(quadcontourset)
    clb.ax.set_ylabel('-overlap', rotation=270, fontsize=15, labelpad=15)
    plt.tight_layout()
    print(f"Saved at {fname}")
    plt.savefig(fname)
