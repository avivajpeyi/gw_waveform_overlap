"""

A file for maximising overlaps for the pycentricity package.

"""

import bilby
import bilby.gw.utils as utils
import numpy as np
from matplotlib import pyplot as plt

from .waveform import Waveform, plot_multiple_waveform_objects

POL = ['cross', 'plus']


def get_zero_noise_psd():
    ifos = bilby.gw.detector.InterferometerList(['H1'])
    ifos.set_strain_data_from_zero_noise(sampling_frequency=2048, duration=32)
    return ifos[0].power_spectral_density


def compute_overlap(wf1: Waveform, wf2: Waveform, psd=None):
    if psd is None:
        psd = get_zero_noise_psd()

    psd_interp = psd.power_spectral_density_interpolated(wf1.frequency)

    # Defining temporary arrays to use
    _psd_interp = psd_interp[wf1.min_fidx:wf1.max_fidx]
    _a = {
        key: wf1.frequency_domain_signal[key][wf1.min_fidx:wf1.max_fidx] for key in POL
    }
    _b = {
        key: wf2.frequency_domain_signal[key][wf1.min_fidx:wf1.max_fidx] for key in POL
    }
    # Doing the calculation
    inner_a = utils.noise_weighted_inner_product(
        _a["plus"], _a["plus"], _psd_interp, wf1.duration
    )
    inner_a += utils.noise_weighted_inner_product(
        _a["cross"], _a["cross"], _psd_interp, wf1.duration
    )
    inner_b = utils.noise_weighted_inner_product(
        _b["plus"], _b["plus"], _psd_interp, wf1.duration
    )
    inner_b += utils.noise_weighted_inner_product(
        _b["cross"], _b["cross"], _psd_interp, wf1.duration
    )
    inner_ab = utils.noise_weighted_inner_product(
        _a["plus"], _b["plus"], _psd_interp, wf1.duration
    )
    inner_ab += utils.noise_weighted_inner_product(
        _a["cross"], _b["cross"], _psd_interp, wf1.duration
    )
    overlap = inner_ab / np.sqrt(inner_a * inner_b)
    return overlap.real


def plot_overlap(wf1, wf2, psd=None, filename=None):
    overlap = compute_overlap(wf1, wf2, psd)
    axes = plot_multiple_waveform_objects(waveform_objects=[wf1, wf2], freq_domain=True)
    axes[0].set_title(f"Overlap = {overlap:.2f}")
    if filename:
        plt.tight_layout()
        plt.savefig(filename)
    return axes