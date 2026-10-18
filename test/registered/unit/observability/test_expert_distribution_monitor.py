"""CPU contract tests plus a real GPU graph replay test when available."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.observability.expert_distribution_monitor import (
    ExpertDistributionMonitor,
    MonitorConfig,
)


class TestExpertDistributionMonitor(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = SimpleNamespace(wall=1000.0, mono=100.0)
        self.addCleanup(patch.stopall)
        patch(
            "sglang.srt.observability.expert_distribution_monitor.time.time",
            side_effect=lambda: self.clock.wall,
        ).start()
        patch(
            "sglang.srt.observability.expert_distribution_monitor.time.monotonic",
            side_effect=lambda: self.clock.mono,
        ).start()

    def monitor(self, device="cpu", delay=0, rank=0):
        mapping = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        metadata = SimpleNamespace(
            num_layers=2,
            num_local_physical_experts=2,
            num_logical_experts=4,
            physical_to_logical_map=mapping.to(device),
            physical_to_logical_map_cpu=mapping.clone(),
        )
        return ExpertDistributionMonitor(
            MonitorConfig(True, 10, delay, self.directory.name),
            metadata,
            device=device,
            global_rank=rank,
            topology={"ep_rank": rank, "ep_size": 2},
        )

    def step(self, m, counts=(2, 3), eligible=True):
        m.begin_forward(eligible)
        with m.with_layer(1):
            m.on_dispatch(torch.tensor(counts, device=m.scratch.device))
        m.end_forward()

    def test_config_disabled_and_invalid_duration(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_EXPERT_DIST_ENABLE": "0",
                "SGLANG_EXPERT_DIST_DURATION_SEC": "bad",
            },
        ):
            self.assertFalse(MonitorConfig.from_env().enabled)
        for value in ("nan", "inf", "0", "-1", "bad"):
            with patch.dict(
                os.environ,
                {
                    "SGLANG_EXPERT_DIST_ENABLE": "1",
                    "SGLANG_EXPERT_DIST_DURATION_SEC": value,
                },
            ):
                with self.assertRaises(ValueError):
                    MonitorConfig.from_env()

    def test_delayed_window_idle_export_and_one_shot(self):
        m = self.monitor(delay=2)
        self.step(m)  # Startup / capture never starts the timer.
        self.assertEqual(m.steps, 0)
        m.arm("abc123", 1000)
        self.step(m)
        self.assertEqual(m.state, "DELAY")
        self.clock.wall, self.clock.mono = 1002, 102
        self.step(m)
        self.assertEqual(m.steps, 1)
        self.clock.wall, self.clock.mono = 1012, 112
        m.poll()  # No more forward calls are needed to finish.
        self.assertEqual(m.state, "DONE")
        with open(m.output_file) as f:
            output = json.load(f)
        self.assertEqual(output["physical_count"][1], [2, 3])
        self.assertEqual(output["logical_count"][1], [2, 3, 0, 0])
        with self.assertRaises(ValueError):
            m.arm("def456", 1012)
        self.step(m)
        self.assertEqual(m.steps, 1)

    def test_layout_change_uses_mapping_of_each_forward(self):
        m = self.monitor()
        m.arm("abc123", 1000)
        self.step(m)
        for mapping in (
            m.metadata.physical_to_logical_map,
            m.metadata.physical_to_logical_map_cpu,
        ):
            mapping[1, :2] = torch.tensor([1, 0])
        self.step(m)
        self.assertEqual(m.logical[1].tolist(), [5, 5, 0, 0])
        self.assertEqual(m.physical[1].tolist(), [4, 6])
        self.assertEqual(len(m.layout_changes), 2)
        self.assertEqual(m.logical.sum(), m.physical.sum())

    def test_draft_padding_and_out_of_window_do_not_accumulate(self):
        m = self.monitor()
        m.arm("abc123", 1000)
        m.begin_forward(True)
        with m.with_layer(1):
            with m.disable():
                m.on_dispatch(torch.tensor([100, 100]))
            m.on_dispatch(torch.tensor([0, 3]))
        m.end_forward()
        self.step(m, eligible=False)
        self.assertEqual(m.physical[1].tolist(), [0, 3])
        self.assertEqual(m.steps, 1)

    def test_whole_forward_admitted_before_deadline(self):
        m = self.monitor()
        m.arm("abc123", 1000)
        self.clock.wall, self.clock.mono = 1009, 109
        m.begin_forward(True)
        with m.with_layer(1):
            m.on_dispatch(torch.tensor([4, 5]))
        self.clock.wall, self.clock.mono = 1011, 111
        m.end_forward()
        self.assertEqual(m.state, "DONE")
        self.assertEqual(m.physical[1].tolist(), [4, 5])

    def test_nonzero_ep_rank_maps_local_slots(self):
        m = self.monitor(rank=1)
        m.arm("abc123", 1000)
        self.step(m)
        self.assertEqual(m.logical[1].tolist(), [0, 0, 2, 3])

    def test_summary_requires_complete_ranks_and_preserves_totals(self):
        from sglang.srt.observability.expert_distribution_summary import summarize

        monitors = [self.monitor(rank=r) for r in range(2)]
        for m in monitors:
            m.arm("abc123", 1000)
            self.step(m)
        self.clock.wall, self.clock.mono = 1010, 110
        monitors[0].poll()
        directory = self.directory.name + "/abc123"
        with self.assertRaises(ValueError):
            summarize(directory)
        monitors[1].poll()
        result = summarize(directory)
        self.assertTrue(result["complete"])
        self.assertEqual(result["assignments"], 10)
        self.assertEqual(result["layers"][1]["max_over_mean"], 1)

    def test_recorder_facade_keeps_monitor_independent(self):
        from sglang.srt.eplb.expert_distribution import (
            _ExpertDistributionRecorderNoop,
            _ExpertDistributionRecorderReal,
        )
        from sglang.srt.observability.expert_distribution_monitor import set_monitor
        from sglang.srt.utils import Withable

        m = self.monitor()
        set_monitor(m)
        self.addCleanup(set_monitor, None)
        m.arm("abc123", 1000)
        noop = _ExpertDistributionRecorderNoop()
        real = object.__new__(_ExpertDistributionRecorderReal)
        real._current_layer_idx = Withable()
        real._disable_all = False
        real._recording = False  # A stopped EPLB recorder must not stop monitor.
        for recorder in (noop, real):
            m.begin_forward(True)
            with recorder.with_current_layer(1):
                with recorder.disable_this_region():
                    recorder.on_deepep_dispatch_low_latency(torch.tensor([99, 99]))
                recorder.on_deepep_dispatch_low_latency(torch.tensor([2, 3]))
            m.end_forward()
        self.assertEqual(m.physical[1].tolist(), [4, 6])

    @unittest.skipUnless(torch.cuda.is_available(), "GPU graph runtime required")
    def test_graph_captured_before_arm_replays_each_step(self):
        m = self.monitor(device="cuda")
        counts = torch.tensor([2, 3], device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with m.with_layer(1):
                m.on_dispatch(counts)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with m.with_layer(1):
                m.on_dispatch(counts)
        self.assertEqual(m.steps, 0)
        m.arm("abc123", 1000)
        for _ in range(3):
            m.begin_forward(True)
            graph.replay()
            m.end_forward(graph=True)
        self.assertEqual(m.physical[1].tolist(), [6, 9])
        self.assertEqual(m.graph_steps, 3)


if __name__ == "__main__":
    unittest.main()
