import copy
import os
import unittest

import numpy as np
from matplotlib import pyplot as plt

from gw_waveform_overlapper import overlap_computer
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
            geocent_time=2,
            ra=0.9978,
            dec=-0.4476
        )
        self.params2 = self.params.copy()
        self.wf1 = Waveform.inject_signal(self.params)
        self.wf2 = Waveform.inject_signal(self.params2)
        self.approximant = "IMRPheonomPv2"
        self.outdir = "tests/overlap_optimizer_test"
        os.makedirs(self.outdir, exist_ok=True)

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_maximiser_for_identical_waveform(self):
        self.wf2 = Waveform.inject_signal(self.params2)
        time_shift, phase_shift = 0, 0
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        res = overlap_optimizer.fast_overlap_optimize(self.wf1, self.wf2, verbose=True)
        self.check_results([time_shift, phase_shift, overlap], res)

    def test_maximiser_for_zero_identical_waveform(self):
        self.wf2 = Waveform.inject_signal(self.params2)
        time_shift, phase_shift = 0, 0
        self.wf2.time_shift(time_shift)
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        res = overlap_optimizer.fast_overlap_optimize(self.wf1, self.wf2, verbose=True)
        plot_waveform_optimization(self.wf1, self.wf2, res[3],
                                   fname=os.path.join(self.outdir, "identical.png"))
        self.check_results([time_shift, phase_shift, overlap], res)

    def test_maximiser_for_time_difference(self):
        self.wf2 = Waveform.inject_signal(self.params2)
        time_shift, phase_shift = 2, 0
        self.wf2.time_shift(time_shift)
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        res = overlap_optimizer.fast_overlap_optimize(self.wf1, self.wf2, verbose=True)
        plot_waveform_optimization(self.wf1, self.wf2, res[3],
                                   fname=os.path.join(self.outdir, "timediff.png"))
        self.check_results([time_shift, phase_shift, overlap], res)

    def test_maximiser_for_phase_difference(self):
        self.wf2 = Waveform.inject_signal(self.params2)
        time_shift, phase_shift = 0, -np.pi / 4
        self.wf2.phase_shift(phase_shift)
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        res = overlap_optimizer.fast_overlap_optimize(self.wf1, self.wf2, verbose=True)
        plot_waveform_optimization(self.wf1, self.wf2, res[3],
                                   fname=os.path.join(self.outdir, "phasediff.png"))
        self.check_results([time_shift, phase_shift, overlap], res)

    def test_maximiser_for_both_difference(self):
        self.wf2 = Waveform.inject_signal(self.params2)
        time_shift, phase_shift = 0.3, -np.pi / 4
        self.wf2.time_shift(time_shift)
        overlap = overlap_optimizer.overlap_computer.compute_overlap(self.wf1, self.wf2)
        res = overlap_optimizer.fast_overlap_optimize(self.wf1, self.wf2, verbose=True)
        plot_waveform_optimization(self.wf1, self.wf2, res[3],
                                   fname=os.path.join(self.outdir, "bothdiff.png"))
        self.check_results([time_shift, phase_shift, overlap], res)

    def check_results(self, orig, res):
        original_timeshift, original_phase, orig_overlap = orig[0], orig[1], orig[2]
        calc_timeshift, calc_phase, calc_overlap = res[0], res[1], res[2]
        # self.assertAlmostEqual(
        #     original_timeshift, calc_timeshift, delta=0.1,
        #     msg=f"Time: ({original_timeshift:.2f}, {calc_timeshift:.2f})"
        # )
        # self.assertAlmostEqual(
        #     original_phase, calc_phase, delta=0.1,
        #     msg=f"Phase: ({original_phase:.2f}, {calc_phase:.2f})"
        # )

        if not (np.greater_equal(calc_overlap, orig_overlap) or
                np.isclose(calc_overlap, orig_overlap, rtol=0.1)
        ):
            self.fail(msg=f"Overlap: ({orig_overlap:.2f}, {calc_overlap:.2f})")


def plot_waveform_optimization(wf1, wf2, path, fname):
    path = np.array(path).T

    t, p = np.meshgrid(
        np.linspace(*overlap_optimizer.TLIM, num=25),
        np.linspace(*overlap_optimizer.PLIM, num=25)
    )

    @np.vectorize
    def f(t, p):
        temp_wf2 = copy.deepcopy(wf2)
        temp_wf2.time_shift(t)
        temp_wf2.phase_shift(p)
        return -overlap_computer.compute_overlap(wf1, temp_wf2)

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
    ax.set_xlim(*overlap_optimizer.TLIM)
    ax.set_ylim(*overlap_optimizer.PLIM)
    clb = fig.colorbar(quadcontourset)
    clb.ax.set_ylabel('-overlap', rotation=270, fontsize=15, labelpad=15)
    plt.tight_layout()
    print(f"Saved at {fname}")
    plt.savefig(fname)


if __name__ == '__main__':
    unittest.main()
