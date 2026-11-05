"""Opt-in source-side DCP packing. Destination KV layout/protocol is unchanged."""

import numpy as np
import time
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(PTRS, INDICES, SIZES, OFFSETS, OUT, N, BLOCK: tl.constexpr):
    layer = tl.program_id(0)
    item = tl.load(SIZES + layer)
    base = tl.load(PTRS + layer).to(tl.pointer_type(tl.uint8))
    offset = tl.load(OFFSETS + layer)
    x = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = x < N * item
    token = tl.load(INDICES + x // item, mask=mask, other=0)
    value = tl.load(base + token * item + x % item, mask=mask, other=0)
    tl.store(OUT + offset + x, value, mask=mask)


def packed_transfer_addresses(dst_indices, dst_ptrs, item_lens, packed_ptr):
    """Only split at discontinuities in destination physical token positions."""
    dst = np.asarray(dst_indices, dtype=np.int64)
    if dst.ndim != 1 or len(dst_ptrs) != len(item_lens):
        raise ValueError("Invalid DCP destination geometry")
    if not dst.size:
        return [], [], []
    if np.any(dst < 0) or any(size <= 0 for size in item_lens):
        raise ValueError("Negative token index or nonpositive token byte size")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(dst) != 1) + 1))
    counts = np.diff(np.append(starts, dst.size))
    srcs, dsts, lengths = [], [], []
    offset = 0
    for ptr, size in zip(dst_ptrs, item_lens):
        srcs.extend((packed_ptr + offset + starts * size).tolist())
        dsts.extend((ptr + dst[starts] * size).tolist())
        lengths.extend((counts * size).tolist())
        offset += dst.size * size
    return srcs, dsts, lengths


def iter_packed_transfer_slices(num_tokens, item_lens, capacity_bytes):
    """Partition token rows; every batch includes the same set of KV layers."""
    if num_tokens < 0 or capacity_bytes <= 0:
        raise ValueError("Invalid DCP token count or packing capacity")
    if any(size <= 0 for size in item_lens):
        raise ValueError("DCP token byte sizes must be positive")
    if not num_tokens or not item_lens:
        return
    tokens_per_batch = capacity_bytes // sum(item_lens)
    if not tokens_per_batch:
        raise ValueError("DCP packing buffer cannot hold one token across all layers")
    for start in range(0, num_tokens, tokens_per_batch):
        yield slice(start, min(start + tokens_per_batch, num_tokens))


class DCPPackedTransfer:
    """One reusable, registered GPU buffer per transfer worker, with a fixed cap.

    A worker must retain exclusive ownership until batch_transfer_sync returns.
    The owning manager must retain this object for the engine's entire lifetime.
    GPU pointers/indices are consumed as raw bytes, including fp8 scale bytes.
    """

    def __init__(self, engine, gpu_id, capacity_bytes):
        self.engine = engine
        self.device = torch.device("cuda", gpu_id)
        self.capacity_bytes = capacity_bytes
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.buffer = torch.empty(capacity_bytes, dtype=torch.uint8, device=self.device)
            ret = engine.batch_register([self.buffer.data_ptr()], [capacity_bytes])
            if ret not in (None, 0):
                raise RuntimeError(f"DCP packing buffer registration failed: {ret}")

    def send(self, session, src_indices, dst_indices, src_ptrs, dst_ptrs, item_lens):
        src_indices = np.asarray(src_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_indices, dtype=np.int64)
        n = src_indices.size
        if (
            src_indices.ndim != 1 or dst_indices.ndim != 1
            or dst_indices.size != n
            or not (len(src_ptrs) == len(dst_ptrs) == len(item_lens))
        ):
            raise ValueError("Mismatched DCP packing geometry")
        # Validate the complete plan before the first write, including its tail.
        if (
            self.capacity_bytes <= 0
            or any(size <= 0 for size in item_lens)
            or np.any(src_indices < 0) or np.any(dst_indices < 0)
        ):
            raise ValueError("Invalid DCP token index, token byte size or capacity")
        if n and item_lens and sum(item_lens) > self.capacity_bytes:
            raise ValueError("DCP packing buffer cannot hold one token across all layers")
        totals = {
            "pack_ms": 0.0, "blocks_ms": 0.0, "engine_ms": 0.0,
            "batches": 0, "max_batch_bytes": 0, "completed_tokens": 0,
        }
        self.last_stats = totals
        if not n or not src_ptrs:
            return 0, 0
        blocks = 0
        for part in iter_packed_transfer_slices(n, item_lens, self.capacity_bytes):
            # _send_batch synchronizes GPU packing and then waits for RDMA.
            # Only a completed batch may be overwritten by the next batch.
            ret, batch_blocks = self._send_batch(
                session, src_indices[part], dst_indices[part],
                src_ptrs, dst_ptrs, item_lens,
            )
            stats = self.last_stats
            for key in ("pack_ms", "blocks_ms", "engine_ms"):
                totals[key] += stats[key]
            totals["batches"] += 1
            totals["max_batch_bytes"] = max(
                totals["max_batch_bytes"], (part.stop - part.start) * sum(item_lens),
            )
            blocks += batch_blocks
            self.last_stats = totals
            if ret != 0:
                return ret, blocks
            totals["completed_tokens"] += part.stop - part.start
        return 0, blocks

    def _send_batch(self, session, src_indices, dst_indices, src_ptrs, dst_ptrs, item_lens):
        n = len(src_indices)
        if n * sum(item_lens) > self.capacity_bytes:
            raise ValueError("Internal DCP batch exceeds packing capacity")
        offsets = np.cumsum([0] + [n * size for size in item_lens[:-1]], dtype=np.int64)
        start = time.perf_counter()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            # Keep all metadata tensors alive until this stream has completed.
            ptrs = torch.tensor(src_ptrs, dtype=torch.int64, device=self.device)
            indices = torch.tensor(np.asarray(src_indices), dtype=torch.int64, device=self.device)
            sizes = torch.tensor(item_lens, dtype=torch.int64, device=self.device)
            offsets_gpu = torch.tensor(offsets, dtype=torch.int64, device=self.device)
            _pack_kernel[(len(src_ptrs), triton.cdiv(n * max(item_lens), 4096))](
                ptrs, indices, sizes, offsets_gpu, self.buffer, n, BLOCK=4096,
            )
            self.stream.synchronize()
        packed_at = time.perf_counter()
        srcs, dsts, lengths = packed_transfer_addresses(
            dst_indices, dst_ptrs, item_lens, self.buffer.data_ptr(),
        )
        # Never fall back/reuse the buffer before the synchronous call completes.
        engine_at = time.perf_counter()
        ret = self.engine.batch_transfer_sync(session, srcs, dsts, lengths)
        self.last_stats = {
            "pack_ms": (packed_at - start) * 1000,
            "blocks_ms": (engine_at - packed_at) * 1000,
            "engine_ms": (time.perf_counter() - engine_at) * 1000,
        }
        return ret, len(lengths)
