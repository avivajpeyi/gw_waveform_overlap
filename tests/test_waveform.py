import os
import unittest
import shutil
import matplotlib.pyplot as plt

from gw_waveform_overlapper.waveform import Waveform, plot_multiple_waveform_objects


class WaveformTest(unittest.TestCase):

    def setUp(self):
        self.params = dict(
            mass_1=141.0741,
            mass_2=113.0013,
            a_1=0.9434,
            a_2=0.2173,
            tilt_1=0,
            tilt_2=0,
            phi_jl=0,
            phi_12=0,
            luminosity_distance=1782.1610,
            theta_jn=0.9614,
            psi=1.6831,
            phase=5.2220,
            geocent_time=1242424473.5880,
            ra=0.9978,
            dec=-0.4476
        )
        self.params2 = self.params.copy()
        self.params2.update(dict(mass_2=20))
        self.approximant = "IMRPheonomPv2"
        self.outdir = "tests/waveform_test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_waveform_constructor(self):
        wf = Waveform.inject_signal(self.params)
        self.assertIsNotNone(wf)

    def test_waveform_plotter(self):
        wf = Waveform.inject_signal(self.params)
        wf2 = Waveform.inject_signal(self.params2)
        ax = wf.plot_time_domain_data(label="Waveform1")
        ax = wf2.plot_time_domain_data(ax, label="Waveform2")
        ax.set_xlim(1.5, 2.5)
        path = os.path.join(self.outdir, "waveforms.png")
        plt.savefig(path)
        self.assertTrue(os.path.exists(path))

    def test_freq_waveform_plotter(self):
        wf = Waveform.inject_signal(self.params)
        wf2 = Waveform.inject_signal(self.params2)
        fig, axes = plt.subplots(2, 1)
        axes[0] = wf.plot_time_domain_data(axes[0], label="Waveform1")
        axes[0] = wf2.plot_time_domain_data(axes[0], label="Waveform2")
        axes[0].set_xlim(1.5, 2.5)
        axes[1] = wf.plot_frequency_domain_data(axes[1], label="Waveform1")
        axes[1] = wf2.plot_frequency_domain_data(axes[1], label="Waveform2")

        path = os.path.join(self.outdir, "freq_waveforms.png")
        plt.savefig(path)
        self.assertTrue(os.path.exists(path))

    def test_plot_multiple_waveform_objects(self):
        path = os.path.join(self.outdir, "combined_waveforms.png")
        plot_multiple_waveform_objects(
            waveform_objects=[
                Waveform.inject_signal(self.params),
                Waveform.inject_signal(self.params2)
            ],
            freq_domain=True,
            filename=path
        )
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
