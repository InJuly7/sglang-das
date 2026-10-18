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
        return cls(
            True,
            duration,
            delay,
            os.environ.get("SGLANG_EXPERT_DIST_OUTPUT_DIR", "expert_distribution"),
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
            self.logical.scatter_add_(1, mapping.long(), self.scratch)
            if self.scratch.is_cuda:
                if self._last_event is None:
                    self._last_event = torch.cuda.Event()
                self._last_event.record(torch.cuda.current_stream())
            self.steps += 1
            self.graph_steps += int(graph)
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
            path = directory / f"rank-{self.global_rank}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(output))
            temporary.replace(path)
            self.output_file = str(path.resolve())
            self.state = "DONE"
            logger.info(
                "[ExpertMonitor] rank=%s DONE steps=%s graph_steps=%s file=%s",
                self.global_rank,
                self.steps,
                self.graph_steps,
                self.output_file,
            )
        except Exception as exc:
            self.error = str(exc)
            self.state = "FAILED"
            logger.exception("[ExpertMonitor] export failed")
