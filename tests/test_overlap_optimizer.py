import os
import shutil
import unittest

import numpy as np

from gw_waveform_overlapper import overlap_optimizer
from gw_waveform_overlapper.waveform import Waveform


class OverlapOptimizerTest(unittest.TestCase):

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
        self.wf1 = Waveform.inject_signal(self.params)
        self.wf2 = Waveform.inject_signal(self.params2)
        self.approximant = "IMRPheonomPv2"
        self.outdir = "tests/overlap_optimizer_test"
        self.adjust_constants()
        os.makedirs(self.outdir, exist_ok=True)

    def adjust_constants(self):
        overlap_optimizer.TLIM = (-0.5, 0.5)
        overlap_optimizer.PLIM = (0, 2*np.pi)
        overlap_optimizer.MAX_ITR = 2
        overlap_optimizer.TOL = 1e-10
        overlap_optimizer.NITER = 1
        overlap_optimizer.NUM = 25

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_maximiser_for_identical_waveform(self):
        fname = "identical.png"
        kwargs = dict(time_shift=0, phase_shift=0)
        self.maximiser_test(**kwargs, fname=fname)
        self.maximiser_test(**kwargs, fname=fname.replace(".", "_fft."),
                            optimizer_method=overlap_optimizer.fft_overlap_optimizer)

    def test_maximiser_for_time_difference(self):
        fname = "tdiff.png"
        kwargs = dict(time_shift=2.0, phase_shift=0)
        self.maximiser_test(**kwargs, fname=fname)
        self.maximiser_test(**kwargs, fname=fname.replace(".", "_fft."),
                            optimizer_method=overlap_optimizer.fft_overlap_optimizer)

    def test_maximiser_for_phase_difference(self):
        fname = "pdif.png"
        kwargs = dict(time_shift=0.0, phase_shift=-np.pi / 4)
        self.maximiser_test(**kwargs, fname=fname)
        self.maximiser_test(**kwargs, fname=fname.replace(".", "_fft."),
                            optimizer_method=overlap_optimizer.fft_overlap_optimizer)

    def test_maximiser_for_both_difference(self):
        fname = "bothdiff.png"
        kwargs = dict(time_shift=0, phase_shift=0)
        # self.maximiser_test(**kwargs, fname=fname, optimizer_method=overlap_optimizer.fft_overlap_optimizer)
        self.maximiser_test(**kwargs, fname=fname.replace(".", "_fft."),
                            optimizer_method=overlap_optimizer.fft_overlap_optimizer)

    def maximiser_test(self, time_shift, phase_shift, fname, optimizer_method):
        self.wf2 = Waveform.inject_signal(self.params)
        self.wf2.time_shift(time_shift)
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        overlap_optimizer.overlap_computer._unpack_data(self.wf1, self.wf2)
        res = optimizer_method(self.wf1, self.wf2, verbose=True)
        overlap_optimizer.plot_waveform_optimization(
            self.wf1, self.wf2, res[3], fname=os.path.join(self.outdir, fname)
        )
        self.check_results([time_shift, phase_shift, overlap, []], res)

    def check_results(self, orig, res):
        msg = f"orig: {[f'{v:.1f}' for v in orig[:-1]]}, new: {[f'{v:.1f}' for v in res[:-1]]})"
        print(msg)
        if not (np.greater_equal(np.abs(res[2]), np.abs(orig[2])) or
                np.isclose(np.abs(res[2]), np.abs(orig[2]), rtol=0.1)
        ):
            self.fail()


if __name__ == '__main__':
    unittest.main()
