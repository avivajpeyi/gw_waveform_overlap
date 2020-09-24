"""

A file for generating waveforms and related objects for the pycentricity package.

"""

import bilby
import bilby.gw.utils as gwutils
import matplotlib.pyplot as plt
import numpy as np
from bilby.gw.detector.strain_data import InterferometerStrainData
from matplotlib.ticker import (AutoMinorLocator)
from copy import deepcopy

STRAIN_LABEL = r'Strain [strain/$\sqrt{\rm Hz}$]'
TIME_LABEL = r'Time (s)'
FREQ_LABEL = r'Frequency [Hz]'
POLARISATION = ['cross', 'plus']
DEFAULT_SAMPLING_FREQ = 2048


class Waveform:
    def __init__(self, time, time_domain_signal, frequency, frequency_domain_signal,
                 approximant, parameters, sampling_frequency):
        """

        :param time: ndarray of time
        :param signal: dict of 'cross' and 'plus' signal data
        :param approximant: str of the approximant for the signal
        :param parameters: dict of params
        """
        self.time = time
        self.time_domain_signal = time_domain_signal
        self.frequency = frequency
        self.frequency_domain_signal = frequency_domain_signal
        self.approximant = approximant
        self.parameters = parameters
        self.strain = InterferometerStrainData()
        self.strain.set_from_frequency_domain_strain(
            frequency_domain_strain=frequency_domain_signal['cross'],
            frequency_array=frequency
        )
        self.duration = max(time)
        self.min_fidx = np.where(self.frequency >= self.strain.minimum_frequency)[0][0]
        self.max_fidx = np.where(self.frequency >= self.strain.maximum_frequency)[0][0]
        self.sampling_frequency = sampling_frequency
        # update time domain signal
        self.set_time_domain_signal_from_frequency()

    @classmethod
    def inject_signal(cls, injection_parameters, approximant='IMRPhenomPv2', duration=4,
                      sampling_frequency=DEFAULT_SAMPLING_FREQ, reference_frequency=50):
        """Generates strain and time data for a set of injection parameters"""
        generator_args = dict(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameters=injection_parameters,
            waveform_arguments=dict(
                reference_frequency=reference_frequency,
                waveform_approximant=approximant
            )
        )

        generator = bilby.gw.WaveformGenerator(**generator_args)
        signal = generator.time_domain_strain(injection_parameters)
        freq_signal = generator.frequency_domain_strain(injection_parameters)

        return cls(
            time=generator.time_array,
            time_domain_signal=signal,
            frequency=generator.frequency_array,
            frequency_domain_signal=freq_signal,
            approximant=approximant,
            parameters=injection_parameters,
            sampling_frequency=sampling_frequency
        )

    def time_shift(self, amount):
        # time shift
        shift_factor = -2j * np.pi * (self.duration + amount) * self.frequency
        self.frequency_domain_signal = {
            key: self.frequency_domain_signal[key] * np.exp(shift_factor)
            for key in POLARISATION
        }
        self.set_time_domain_signal_from_frequency()

    def set_time_domain_signal_from_frequency(self):
        self.time_domain_signal = {
            key: bilby.core.utils.infft(
                self.frequency_domain_signal[key], self.sampling_frequency)
            for key in POLARISATION
        }

    def phase_shift(self, amount):
        # phase shift
        self.frequency_domain_signal = {
            key: self.frequency_domain_signal[key] * np.exp(-2j * amount)
            for key in POLARISATION
        }
        self.set_time_domain_signal_from_frequency()

    def plot_time_domain_data(self, ax=None, label=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(STRAIN_LABEL)
        ax.set_xlabel(TIME_LABEL)

        if label is None:
            label = self.approximant

        signal = self.time_domain_signal['cross']  # has max val at end
        signal = np.roll(signal, shift=len(signal) // 2)  # move max val to mid
        if color:
            ax.plot(self.time, signal, label=label, color=color)
        else:
            ax.plot(self.time, signal, label=label)
        ax.set_xlim(left=1.5, right=2.5)
        return ax

    def plot_frequency_domain_data(self, ax=None, label=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(STRAIN_LABEL)
        ax.set_xlabel(FREQ_LABEL)
        if label is None:
            label = self.approximant

        freq_data = self.frequency_domain_signal['cross']
        f = self.frequency
        df = f[1] - f[0]

        asd = gwutils.asd_from_freq_series(
            freq_data=freq_data, df=df)
        if color:
            ax.loglog(self.strain.frequency_array[self.strain.frequency_mask],
                      asd[self.strain.frequency_mask], label=label, color=color)
        else:
            ax.loglog(self.strain.frequency_array[self.strain.frequency_mask],
                      asd[self.strain.frequency_mask], label=label)
        ax.set_xlim(left=10, right=300)
        return ax

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v))
        return result


def plot_multiple_waveform_objects(waveform_objects, freq_domain=False,
                                   filename=None):
    if freq_domain:
        fig, axes = plt.subplots(2, 1)
        time_ax = axes[0]
        freq_ax = axes[1]
    else:
        fig, time_ax = plt.subplots()

    for i, wf in enumerate(waveform_objects):
        time_ax = wf.plot_time_domain_data(time_ax, label=f"Waveform {i}")

    if freq_domain:
        for i, wf in enumerate(waveform_objects):
            freq_ax = wf.plot_frequency_domain_data(freq_ax, label=f"Waveform {i}")

    time_ax.legend(loc='best')

    if filename:
        plt.tight_layout()
        fig.savefig(filename)
    else:
        return axes
