"""

A file for maximising overlaps for the pycentricity package.

"""

import bilby
import numpy as np
from matplotlib import pyplot as plt

from .waveform import Waveform, plot_multiple_waveform_objects, POLARISATION


def get_zero_noise_psd():
    ifos = bilby.gw.detector.InterferometerList(['H1'])
    ifos.set_strain_data_from_zero_noise(sampling_frequency=2048, duration=32)
    return ifos[0].power_spectral_density


def compute_overlap(wf1: Waveform, wf2: Waveform, psd=None):
    """
    Eq1 https://arxiv.org/pdf/1806.05350.pdf

    where a and b are GW templates (in f domain)

    The overlap between a waveform `a` and another waveform `b`,
    maximizing over the time and phase (t, φ) of the waveform,

    O(a,b) =  max_{t,φ} [<a|b> / sqrt( <a|a> * <b|b> )]

    :param wf1:
    :param wf2:
    :param psd:
    :return: Overlap
        The overlap takes on values between -1 (corresponding to waveforms 180◦
        out of phase) and 1 (for identical waveforms).
    """
    inner_a = inner_product(wf1, wf1)
    inner_b = inner_product(wf2, wf2)
    inner_ab = inner_product(wf1, wf2)
    overlap = inner_ab / np.sqrt(inner_a * inner_b)
    overlap = overlap.real
    if round(overlap, 2) > 1 or round(overlap, 2) < -1:
        raise ValueError(f"Overlap of {overlap} is out of bound [-1, 1]")
    return overlap


def _unpack_data(wf1, wf2, psd=None):
    if psd is None:
        psd = get_zero_noise_psd()
    freq, dur = wf1.frequency, wf1.duration

    psd_interp = psd.power_spectral_density_interpolated(freq)
    a = {p: wf1.frequency_domain_signal[p] for p in POLARISATION}
    b = {p: wf2.frequency_domain_signal[p] for p in POLARISATION}

    # Doing the calculation
    a = a["plus"] + a["cross"]
    b = b["plus"] + b["cross"]

    lens = [len(k) for k in [a, b, freq, psd_interp] ]
    assert len(set(lens)) == 1, f"a, b, freq, psd_interp = {lens}"
    return a, b, freq, dur, psd_interp


def inner_product(wf_a, wf_b, psd=None):
    """
    :return: (4/duration) Σ [a*(f) b(f) / PSD]
    """
    a, b, freq, dur, psd = _unpack_data(wf_a, wf_b, psd)
    integrand = np.conj(a) * b / psd
    return 4 / dur * np.sum(integrand)


def complex_filter(t0, wf_a: Waveform, wf_b: Waveform, psd=None):
    """
    PRECONDITIONS:
    - phase for a(f) and b(f) to 0
    - time t0 for b(f) == 0 (hence b0)

    :returns z(t0) = 4 int [a *b0 * exp(2*pi*i*f*t0) / psd(f)] df
    """
    a, b, freq, dur, psd = _unpack_data(wf_a, wf_b, psd)
    integrand = np.conj(a) * b * np.exp(2 * np.pi * freq * t0 * 1j) / psd
    constant = 4 / dur
    return constant * np.sum(integrand)


def get_snr_for_overlap(overlap):
    """
    Eq3 https://arxiv.org/pdf/1806.05350.pdf

    solve 1 - O > 1/ρ^2 for ρ
    Results:
        ρ<-sqrt(1/(1 - O)) and O<1
        ρ>sqrt(1/(1 - O)) and O<1

    :return:
    """
    return abs(np.sqrt(1 / (1 - overlap)))


def plot_overlap(wf1, wf2, psd=None, filename=None):
    overlap = compute_overlap(wf1, wf2, psd)
    snr = get_snr_for_overlap(overlap)
    axes = plot_multiple_waveform_objects(waveform_objects=[wf1, wf2], freq_domain=True)
    axes[0].set_title(f"Overlap = {overlap:.2f}, max(ρ) > {snr:.2f} ")
    if filename:
        plt.tight_layout()
        plt.savefig(filename)
    return axes
