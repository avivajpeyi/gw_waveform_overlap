# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from gw_waveform_overlapper.overlap_computer import plot_overlap
from gw_waveform_overlapper.waveform import Waveform


def get_injection_params(a_2):
    """

    :param a_2:
    :param distance: in Mpc
    :return:
    """
    return dict(
        mass_1=20,
        mass_2=20,
        a_1=0.4,
        a_2=a_2,
        tilt_1=0.5,
        tilt_2=1,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=400,
        dec=-1.2208,
        ra=1.375,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=0,
    )


def main():
    w1 = Waveform.inject_signal(get_injection_params(a_2=0))
    w2 = Waveform.inject_signal(get_injection_params(a_2=0.2))
    plot_overlap(w1, w2, filename="max_snr_for_different_a2.png")

if __name__ == "__main__":
    main()