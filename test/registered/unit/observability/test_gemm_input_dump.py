"""Check library-boundary metadata and graph replay masks, not capture masks."""

import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.observability.expert_distribution_monitor import (
    ExpertDistributionMonitor,
    MonitorConfig,
    set_monitor,
)
from sglang.srt.observability.gemm_input_dump import (
    masked_gemm_metadata,
    masked_gemm_record,
)


class TestGemmInputDump(unittest.TestCase):
    def args(self, device="cpu", capacity=8):
        return (
            torch.zeros((2, capacity, 4), device=device, dtype=torch.int8),
            torch.ones((2, capacity, 1), device=device),
            torch.zeros((2, 6, 2), device=device, dtype=torch.int8),
            torch.ones((2, 6, 1), device=device),
            torch.empty((2, capacity, 6), device=device, dtype=torch.bfloat16),
            torch.tensor([2, 3], dtype=torch.int32, device=device),
            3,
        )

    def monitor(self, device="cpu"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        mapping = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        m = ExpertDistributionMonitor(
            MonitorConfig(True, 10, 0, directory.name, True, 1),
            SimpleNamespace(
                num_layers=2,
                num_local_physical_experts=2,
                num_logical_experts=4,
                physical_to_logical_map=mapping.to(device),
                physical_to_logical_map_cpu=mapping,
            ),
            device=device,
            global_rank=0,
            topology={"ep_rank": 0, "ep_size": 2},
        )
        return m

    def test_reference_fields_and_masked_equivalents(self):
        args = self.args()
        metadata = masked_gemm_metadata("deepgemm.test", 1, "gate_up", *args)
        record = masked_gemm_record(
            metadata, [2, 3], run_id="abc", rank=0, step=1, logical_ids=[0, 1]
        )
        self.assertEqual(record["valid_m"], 5)
        self.assertEqual(record["pad_m"], 11)
        self.assertEqual(record["expert_counts"], {"0": 2, "1": 3})
        self.assertEqual(
            [record["derived_" + x] for x in ("E", "M", "N", "K")], [2, 8, 6, 4]
        )
        for descriptor in record["tensors"].values():
            self.assertEqual(
                set(descriptor), {"shape", "dtype", "device", "stride", "is_contiguous"}
            )
        self.assertIsNone(record["m_indices_runs"])
        self.assertEqual(record["masked_m_values"], [2, 3])
        with self.assertRaises(ValueError):
            masked_gemm_record(
                metadata, [9, 3], run_id="abc", rank=0, step=1, logical_ids=[0, 1]
            )

    def test_actual_library_wrapper_gets_same_tensors(self):
        from sglang.srt.layers.moe.ep_moe import layer

        m = self.monitor()
        set_monitor(m)
        self.addCleanup(set_monitor, None)
        args = self.args()
        for api in (
            "m_grouped_w4a8_gemm_nt_masked",
            "m_grouped_w4a8_gemm_nt_masked_hipc",
        ):
            with patch.object(layer, api, return_value=args[4]) as library:
                getattr(layer, api + "_wrapper")(*args, layer_id=1, stage="gate_up")
                passed = library.call_args.args
                self.assertIs(passed[0][0], args[0])
                self.assertIs(passed[1][0], args[2])
                self.assertIs(passed[2], args[4])
                self.assertIs(passed[3], args[5])
                self.assertEqual(passed[4], args[6])
        self.assertEqual(len(m.gemm_variants), 2)

    def test_samples_are_bounded_and_stage_totals_match_dispatch(self):
        m = self.monitor()
        args = self.args()
        m.arm("abc", time.time())
        for _ in range(3):
            m.begin_forward(True)
            with m.with_layer(1):
                m.on_dispatch(args[5])
            for stage in ("gate_up", "down"):
                m.on_gemm("deepgemm.test", 1, stage, *args)
            m.end_forward()
        self.assertEqual(len(m.gemm_records), 2)  # First forward only.
        self.assertEqual(len(m.gemm_variants), 2)  # Deduplicated metadata.
        self.assertEqual(m.gemm_total[1, 0].tolist(), [6, 9])
        self.assertEqual(m.gemm_total[1, 1].tolist(), m.physical[1].tolist())
        m.finalize()
        with open(m.output_file) as f:
            result = json.load(f)
        self.assertTrue(result["gemm_vs_dispatch_match"])
        with open(result["gemm_inputs_file"]) as f:
            self.assertEqual(len(f.readlines()), 2)

    def test_mismatched_gemm_counts_cannot_be_summarized_as_complete(self):
        from pathlib import Path
        from sglang.srt.observability.expert_distribution_summary import summarize

        m = self.monitor()
        m.topology["ep_size"] = 1
        args = self.args()
        m.arm("abc", time.time())
        m.begin_forward(True)
        with m.with_layer(1):
            m.on_dispatch(args[5])
        m.on_gemm("deepgemm.test", 1, "gate_up", *args)  # Missing down stage.
        m.end_forward()
        m.finalize()
        with open(m.output_file) as f:
            self.assertFalse(json.load(f)["gemm_vs_dispatch_match"])
        with self.assertRaisesRegex(ValueError, "GEMM/dispatch"):
            summarize(Path(m.output_file).parent)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU graph runtime required")
    def test_replay_selects_actual_variant_and_dynamic_mask(self):
        m = self.monitor(device="cuda")
        args = self.args("cuda", 8)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for stage in ("gate_up", "down"):
                m.on_gemm("deepgemm.test", 1, stage, *args)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            with m.with_layer(1):
                m.on_dispatch(args[5])
            for stage in ("gate_up", "down"):
                m.on_gemm("deepgemm.test", 1, stage, *args)
        # A different captured/warmup variant must not be mistaken for the replay.
        other = self.args("cuda", 16)
        for stage in ("gate_up", "down"):
            m.on_gemm("deepgemm.test", 1, stage, *other)
        args[5].copy_(torch.tensor([4, 1], device="cuda", dtype=torch.int32))
        m.arm("abc", time.time())
        m.begin_forward(True)
        graph.replay()
        m.end_forward(graph=True)
        self.assertEqual(len(m.gemm_records), 2)
        for record in m.gemm_records:
            self.assertEqual(record["derived_M"], 8)
            self.assertEqual(record["masked_m_values"], [4, 1])
        self.assertEqual(m.gemm_total[1, 0].tolist(), m.physical[1].tolist())


if __name__ == "__main__":
    unittest.main()
