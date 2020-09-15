# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import numpy as np

from gw_waveform_overlapper.multiple_overlaps import plot_overlaps
from gw_waveform_overlapper.waveform import Waveform


def get_injection_params(a_2, distance):
    """

    :param a_2:
    :param distance: in Mpc
    :return:
    """
    return dict(
        mass_1=36,
        mass_2=29,
        a_1=0.4,
        a_2=a_2,
        tilt_1=0.5,
        tilt_2=1,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=distance,
        dec=-1.2208,
        ra=1.375,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=0,
    )


def create_waveforms(dist_range):
    w1s, w2s = [], []
    for d in dist_range:
        w1s.append(Waveform.inject_signal(get_injection_params(a_2=0, distance=d)))
        w2s.append(Waveform.inject_signal(get_injection_params(a_2=0.1, distance=d)))
    return w1s, w2s


def main():
    dist_range = np.linspace(300, 1500, num=50)
    w1s, w2s = create_waveforms(dist_range)
    overlap_x_data = dict(label='Distance [Mpc]', data=dist_range)
    plot_overlaps(w1s, w2s, overlap_x_data, filename='distance_overlap.mp4')


if __name__ == "__main__":
    main()
