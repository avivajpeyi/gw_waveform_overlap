# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import bilby
import numpy as np

from gw_waveform_overlapper.multiple_overlaps import plot_overlaps
from gw_waveform_overlapper.waveform import Waveform


def get_injection_params(total_mass, a_2):
    """

    :param a_2:
    :param distance: in Mpc
    :return:
    """

    m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
        mass_ratio=1,
        total_mass=total_mass
    )

    return dict(
        mass_1=m1,
        mass_2=m2,
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


def create_waveforms(mass_range):
    w1s, w2s = [], []
    for m in mass_range:
        w1s.append(Waveform.inject_signal(get_injection_params(a_2=0, total_mass=m)))
        w2s.append(Waveform.inject_signal(get_injection_params(a_2=0.1, total_mass=m)))
    return w1s, w2s


def main():
    mass_range = np.linspace(40, 120, num=50)
    w1s, w2s = create_waveforms(mass_range)
    overlap_x_data = dict(label='Mass [Msun]', data=mass_range)
    plot_overlaps(w1s, w2s, overlap_x_data, filename='mass_overlap.mp4')


if __name__ == "__main__":
    main()
