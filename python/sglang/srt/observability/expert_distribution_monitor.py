"""Independent, bounded-memory decode expert statistics (no EPLB collectives).

The layer scope is shared with the recorder, but buffers and lifecycle are not.
Only per-forward scratch writes are captured; admission and accumulation happen
outside the graph, before EPLB can change the physical-to-logical mapping.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = False
    duration: float = 60.0
    delay: float = 0.0
    output_dir: str = "expert_distribution"
    dump_gemm: bool = False
    gemm_max_steps: int = 1

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("SGLANG_EXPERT_DIST_ENABLE", "0").lower()
        if enabled not in ("0", "1", "false", "true"):
            raise ValueError("SGLANG_EXPERT_DIST_ENABLE must be 0 or 1")
        if enabled in ("0", "false"):
            return cls()
        duration = float(os.environ.get("SGLANG_EXPERT_DIST_DURATION_SEC", "60"))
        delay = float(os.environ.get("SGLANG_EXPERT_DIST_DELAY_SEC", "0"))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                "SGLANG_EXPERT_DIST_DURATION_SEC must be finite and positive"
            )
        if not math.isfinite(delay) or delay < 0:
            raise ValueError(
                "SGLANG_EXPERT_DIST_DELAY_SEC must be finite and nonnegative"
            )
        dump_gemm = os.environ.get("SGLANG_EXPERT_DIST_DUMP_GEMM", "0").lower()
        if dump_gemm not in ("0", "1", "false", "true"):
            raise ValueError("SGLANG_EXPERT_DIST_DUMP_GEMM must be 0 or 1")
        dump_gemm = dump_gemm in ("1", "true")
        max_steps = (
            int(os.environ.get("SGLANG_EXPERT_DIST_GEMM_MAX_STEPS", "1"))
            if dump_gemm
            else 1
        )
        if max_steps <= 0:
            raise ValueError("SGLANG_EXPERT_DIST_GEMM_MAX_STEPS must be positive")
        return cls(
            True,
            duration,
            delay,
            os.environ.get("SGLANG_EXPERT_DIST_OUTPUT_DIR", "expert_distribution"),
            dump_gemm,
            max_steps,
        )


def get_monitor():
    from sglang.srt.runtime_context import get_resources

    return get_resources().expert_distribution_monitor


def set_monitor(monitor):
    from sglang.srt.runtime_context import get_resources

    get_resources().expert_distribution_monitor = monitor


class ExpertDistributionMonitor:
    def __init__(self, config, metadata, *, device, global_rank, topology):
        import torch

        self.config = config
        self.metadata = metadata
        self.global_rank = global_rank
        self.topology = topology
        self.state = "ARMED"
        self.run_id = None
        self.error = None
        self.output_file = None
        self.layer = None
        self.disabled = 0
        self.steps = 0
        self.graph_steps = 0
        self.first_forward_at = None
        self.last_forward_at = None
        self.finalized_at = None
        self.layout_changes = []
        self._last_mapping = None
        self._include = False
        self._last_event = None
        shape = (metadata.num_layers, metadata.num_local_physical_experts)
        self.scratch = torch.zeros(shape, dtype=torch.int64, device=device)
        self.physical = torch.zeros_like(self.scratch)
        self.logical = torch.zeros(
            (metadata.num_layers, metadata.num_logical_experts),
            dtype=torch.int64,
            device=device,
        )

        self.gemm_records = []
        self.gemm_variants = []
        self._gemm_variant_lookup = {}
        self.gemm_sample_steps = 0
        self.gemm_inputs_file = None
        self.gemm_vs_dispatch_match = None
        if config.dump_gemm:
            self.gemm_counts = torch.zeros(
                (shape[0], 2, shape[1]), dtype=torch.int64, device=device
            )
            self.gemm_total = torch.zeros_like(self.gemm_counts)
            self.gemm_tags = torch.full(
                (shape[0], 2), -1, dtype=torch.int64, device=device
            )

    def on_gemm(self, api, layer, stage, a0, a1, b0, b1, output, masked_m, expected_m):
        if not self.config.dump_gemm or self.disabled:
            return
        from sglang.srt.observability.gemm_input_dump import masked_gemm_metadata

        if layer < 0 or stage not in ("gate_up", "down"):
            raise ValueError("GEMM dump requires explicit layer and stage")
        metadata = masked_gemm_metadata(
            api, layer, stage, a0, a1, b0, b1, output, masked_m, expected_m
        )
        key = json.dumps(metadata, sort_keys=True)
        variant = self._gemm_variant_lookup.get(key)
        if variant is None:
            variant = len(self.gemm_variants)
            self._gemm_variant_lookup[key] = variant
            self.gemm_variants.append(metadata)
        stage_index = 0 if stage == "gate_up" else 1
        # Captured GPU writes select the ACTUALLY REPLAYED variant. Never read
        # masked_m on the CPU while CUDA/HIP graph capture is in progress.
        self.gemm_counts[layer, stage_index].copy_(masked_m)
        self.gemm_tags[layer, stage_index].fill_(variant)

    def _sample_gemm(self):
        from sglang.srt.observability.gemm_input_dump import masked_gemm_record

        counts = self.gemm_counts.cpu().tolist()
        tags = self.gemm_tags.cpu().tolist()
        ep_rank = self.topology["ep_rank"]
        n = self.metadata.num_local_physical_experts
        mapping = self.metadata.physical_to_logical_map_cpu
        for layer, stages in enumerate(tags):
            for stage_index, variant in enumerate(stages):
                if variant < 0:
                    continue
                record = masked_gemm_record(
                    self.gemm_variants[variant],
                    counts[layer][stage_index],
                    run_id=self.run_id,
                    rank=ep_rank,
                    step=self.steps,
                    logical_ids=mapping[
                        layer, ep_rank * n : (ep_rank + 1) * n
                    ].tolist(),
                )
                record["global_rank"] = self.global_rank
                self.gemm_records.append(record)
        self.gemm_sample_steps += 1

    @contextmanager
    def with_layer(self, layer):
        previous, self.layer = self.layer, layer
        try:
            yield
        finally:
            self.layer = previous

    @contextmanager
    def disable(self):
        self.disabled += 1
        try:
            yield
        finally:
            self.disabled -= 1

    def on_dispatch(self, counts):
        # Run even while ARMED / DONE: the scratch op must exist in the graph.
        if self.disabled:
            return
        if self.layer is None:
            raise RuntimeError("Expert monitor dispatch has no layer scope")
        if counts.numel() != self.scratch.shape[1]:
            raise RuntimeError("Expert monitor dispatch shape changed")
        self.scratch[self.layer].add_(counts)

    def arm(self, run_id, trigger_at):
        if self.run_id is not None:
            if self.run_id != run_id:
                raise ValueError("Expert monitor is one-shot; restart for another run")
            return self.status()
        if not run_id or any(c not in "0123456789abcdef-" for c in run_id):
            raise ValueError("Invalid monitor run_id")
        if not math.isfinite(trigger_at):
            raise ValueError("Invalid monitor trigger time")
        self.run_id = run_id
        self.start_at = trigger_at + self.config.delay
        # Convert the common wall-clock anchor once; local deadlines are monotonic.
        self.start_mono = time.monotonic() + self.start_at - time.time()
        self.end_mono = self.start_mono + self.config.duration
        self.state = "DELAY" if time.monotonic() < self.start_mono else "RECORDING"
        logger.info(
            "[ExpertMonitor] rank=%s run=%s start=%s duration=%s",
            self.global_rank,
            run_id,
            self.start_at,
            self.config.duration,
        )
        return self.status()

    def poll(self):
        if self.state not in ("DELAY", "RECORDING"):
            return
        now = time.monotonic()
        if now >= self.end_mono:
            self.finalize()
        elif now >= self.start_mono:
            self.state = "RECORDING"

    def begin_forward(self, eligible):
        self.poll()
        self._include = eligible and self.state == "RECORDING"
        self._forward_at = time.time()
        self.scratch.zero_()
        if self.config.dump_gemm:
            self.gemm_counts.zero_()
            self.gemm_tags.fill_(-1)

    def end_forward(self, *, graph=False):
        import torch

        if self._include:
            ep_rank = self.topology["ep_rank"]
            n = self.metadata.num_local_physical_experts
            # Read the CURRENT mapping before EPLB updates it. The same stream
            # orders replay, accumulation and subsequent layout updates.
            mapping = self.metadata.physical_to_logical_map[
                :, ep_rank * n : (ep_rank + 1) * n
            ]
            self.physical.add_(self.scratch)
            if self.config.dump_gemm:
                self.gemm_total.add_(self.gemm_counts)
            self.logical.scatter_add_(1, mapping.long(), self.scratch)
            if self.scratch.is_cuda:
                if self._last_event is None:
                    self._last_event = torch.cuda.Event()
                self._last_event.record(torch.cuda.current_stream())
            self.steps += 1
            self.graph_steps += int(graph)
            if (
                self.config.dump_gemm
                and self.gemm_sample_steps < self.config.gemm_max_steps
            ):
                self._sample_gemm()
            if self.first_forward_at is None:
                self.first_forward_at = self._forward_at
            self.last_forward_at = self._forward_at
            # CPU metadata has no D2H synchronization; only retain layout changes.
            layout = self.metadata.physical_to_logical_map_cpu
            if self._last_mapping is None or not torch.equal(
                layout, self._last_mapping
            ):
                self.layout_changes.append(
                    {"step": self.steps, "mapping": layout.tolist()}
                )
                self._last_mapping = layout.clone()
        self._include = False
        self.poll()

    def status(self):
        return dict(
            state=self.state,
            run_id=self.run_id,
            rank=self.global_rank,
            steps=self.steps,
            graph_steps=self.graph_steps,
            output_file=self.output_file,
            gemm_inputs_file=self.gemm_inputs_file,
            gemm_vs_dispatch_match=self.gemm_vs_dispatch_match,
            error=self.error,
        )

    def finalize(self):
        if self.state not in ("DELAY", "RECORDING"):
            return
        self.state = "FINALIZING"
        try:
            # One boundary copy, no per-layer/per-step D2H or collective.
            if self._last_event is not None:
                self._last_event.synchronize()
            physical = self.physical.cpu().tolist()
            logical = self.logical.cpu().tolist()
            self.finalized_at = time.time()
            output = dict(
                schema_version=1,
                run_id=self.run_id,
                rank=self.global_rank,
                hostname=socket.gethostname(),
                topology=self.topology,
                config=self.config.__dict__,
                start_at=self.start_at,
                end_at=self.start_at + self.config.duration,
                first_forward_at=self.first_forward_at,
                last_forward_at=self.last_forward_at,
                finalized_at=self.finalized_at,
                boundary="whole target forwards admitted by submission time",
                steps=self.steps,
                graph_steps=self.graph_steps,
                physical_count=physical,
                logical_count=logical,
                layout_changes=self.layout_changes,
            )
            directory = Path(self.config.output_dir) / self.run_id
            directory.mkdir(parents=True, exist_ok=True)
            if self.config.dump_gemm:
                gemm_total = self.gemm_total.cpu().tolist()
                match = all(
                    stages[0] == physical[layer] and stages[1] == physical[layer]
                    for layer, stages in enumerate(gemm_total)
                )
                dump_path = directory / f"rank-{self.global_rank}.gemm_inputs.jsonl"
                temporary_dump = dump_path.with_suffix(".jsonl.tmp")
                with temporary_dump.open("w") as f:
                    for record in self.gemm_records:
                        f.write(json.dumps(record) + "\n")
                temporary_dump.replace(dump_path)
                self.gemm_inputs_file = str(dump_path.resolve())
                self.gemm_vs_dispatch_match = match
                output.update(
                    gemm_inputs_file=str(dump_path.resolve()),
                    gemm_sample_steps=self.gemm_sample_steps,
                    gemm_physical_count=gemm_total,
                    gemm_vs_dispatch_match=match,
                )
                if not match:
                    logger.error(
                        "[ExpertMonitor] GEMM/dispatch counts differ on rank %s",
                        self.global_rank,
                    )
            path = directory / f"rank-{self.global_rank}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(output))
            temporary.replace(path)
            self.output_file = str(path.resolve())
            if self.config.dump_gemm and not self.gemm_vs_dispatch_match:
                self.state = "FAILED"
                self.error = "GEMM/dispatch counts differ"
            else:
                self.state = "DONE"
            logger.info(
                "[ExpertMonitor] rank=%s %s steps=%s graph_steps=%s file=%s",
                self.global_rank,
                self.state,
                self.steps,
                self.graph_steps,
                self.output_file,
            )
        except Exception as exc:
            self.error = str(exc)
            self.state = "FAILED"
            logger.exception("[ExpertMonitor] export failed")
