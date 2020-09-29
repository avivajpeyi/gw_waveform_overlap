import os
import shutil
import unittest

from gw_waveform_overlapper.overlap_computer import compute_overlap, Waveform, \
    plot_overlap


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
            geocent_time=0,
            ra=0.9978,
            dec=-0.4476
        )
        self.params2 = self.params.copy()
        self.params2.update(dict(mass_2=20))
        self.wf1 = Waveform.inject_signal(self.params)
        self.wf2 = Waveform.inject_signal(self.params2)
        self.approximant = "IMRPheonomPv2"
        self.outdir = "tests/overlap_test"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_overlap(self):
        self.assertIsNotNone(compute_overlap(self.wf1, self.wf2))
        self.assertEqual(1, compute_overlap(self.wf1, self.wf1))

    def test_overlap_plot(self):
        path = os.path.join(self.outdir, "overlap_different.png")
        plot_overlap(self.wf1, self.wf2, filename=path)
        self.assertTrue(os.path.exists(path))
        path = os.path.join(self.outdir, "overlap_same.png")
        plot_overlap(self.wf1, self.wf1, filename=path)
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
