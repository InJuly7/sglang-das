"""Boundary tests for the scheduler's waiting / running request timeouts.

Both paths are pure bookkeeping over timestamps -- no model, no GPU, no draft
worker -- so they are driven here directly instead of through a server. The
e2e side (503 reaching the client, server stays up) is covered by
scheduler/test_scheduler_control.py.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.disaggregation.utils import DisaggregationMode

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeReq:
    """Must stay hashable: the waiting-timeout path collects drops in a set."""

    def __init__(self, rid, wait_entry=0.0, forward_entry=0.0, is_finished=False):
        self.rid = rid
        self.to_finish = None
        self._finished = is_finished
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=wait_entry,
            forward_entry_time=forward_entry,
            trace_ctx=MagicMock(),
        )

    def finished(self):
        return self._finished


def _req(
    rid: str, *, wait_entry: float = 0.0, forward_entry: float = 0.0, finished=False
):
    return _FakeReq(rid, wait_entry, forward_entry, finished)


def _scheduler(waiting_queue):
    s = Scheduler.__new__(Scheduler)
    s.waiting_queue = waiting_queue
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    return s


class TestWaitingTimeout(CustomTestCase):
    def test_drops_only_reqs_past_the_deadline(self):
        now = time.perf_counter()
        stale = _req("stale", wait_entry=now - 10)
        fresh = _req("fresh", wait_entry=now)
        s = _scheduler([stale, fresh])

        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1.0):
            s._abort_on_waiting_timeout()

        self.assertEqual([r.rid for r in s.waiting_queue], ["fresh"])
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 1)

    def test_unset_entry_time_is_never_dropped(self):
        # 0 is the "not yet stamped" sentinel; the guard is `0 < entry_time`.
        s = _scheduler([_req("unstamped", wait_entry=0.0)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1e-9):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_disabled_timeout_is_a_no_op(self):
        s = _scheduler([_req("stale", wait_entry=time.perf_counter() - 100)])
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0):
            s._abort_on_waiting_timeout()
        self.assertEqual(len(s.waiting_queue), 1)


class TestRunningTimeout(CustomTestCase):
    @staticmethod
    def _batch(reqs):
        return SimpleNamespace(reqs=reqs, is_empty=lambda: not reqs)

    def test_marks_only_stale_unfinished_reqs(self):
        now = time.perf_counter()
        stale = _req("stale", forward_entry=now - 10)
        fresh = _req("fresh", forward_entry=now)
        done = _req("done", forward_entry=now - 10, finished=True)
        s = _scheduler([])

        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([stale, fresh, done]))

        self.assertIsNotNone(stale.to_finish)
        self.assertIsNone(fresh.to_finish)
        self.assertIsNone(done.to_finish, "a finished req must not be aborted")

    def test_unset_forward_entry_time_is_never_marked(self):
        s = _scheduler([])
        req = _req("unstamped", forward_entry=0.0)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1e-9):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)

    def test_empty_batch_and_disabled_timeout_are_no_ops(self):
        s = _scheduler([])
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(1.0):
            s._abort_on_running_timeout(self._batch([]))
        req = _req("stale", forward_entry=time.perf_counter() - 100)
        with envs.SGLANG_REQ_RUNNING_TIMEOUT.override(0):
            s._abort_on_running_timeout(self._batch([req]))
        self.assertIsNone(req.to_finish)



class TestQueuedAbortWithoutBeamCoordinator(CustomTestCase):
    def _scheduler(self, mode):
        s = Scheduler.__new__(Scheduler)
        s.chunked_req = None
        s._pending_chunked_abort_req = None
        s.mm_receiver = None
        s.dllm_config = None
        s.waiting_queue = []
        s.grammar_manager = MagicMock()
        s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
        s.disaggregation_mode = mode
        s._release_aborted_request = MagicMock()
        s.req_to_metadata_buffer_idx_allocator = MagicMock()
        s.tree_cache = MagicMock()
        s.ps = SimpleNamespace(pp_size=1)
        s.running_batch = SimpleNamespace(reqs=[])
        s.last_batch = None
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        s.disagg_prefill_inflight_queue = []
        s.disagg_decode_prealloc_queue = SimpleNamespace(
            abort_matching=MagicMock(), retracted_queue=[]
        )
        s.disagg_decode_transfer_queue = SimpleNamespace(abort_matching=MagicMock())
        return s

    def _queued(self, rid):
        req = _req(rid)
        req.pending_bootstrap = True
        req.mamba_pool_idx = None
        req.http_worker_ipc = "ipc:///tmp/test-tokenizer"
        req.disagg_kv_sender = MagicMock()
        return req

    def test_prefill_prefix_abort_releases_only_matching_request(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        canceled, kept = self._queued("cancel-one"), self._queued("keep-two")
        s.waiting_queue = [canceled, kept]
        self.assertFalse(hasattr(s, "beam_coordinator"))
        with patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer") as release:
            s.abort_request(AbortReq(rid="cancel-"))
        self.assertEqual(s.waiting_queue, [kept])
        s._release_aborted_request.assert_called_once_with(canceled.rid)
        release.assert_called_once_with(canceled, s.req_to_metadata_buffer_idx_allocator)
        canceled.disagg_kv_sender.abort.assert_called_once()
        kept.disagg_kv_sender.abort.assert_not_called()
        message, owner = s.ipc_channels.send_to_tokenizer.send_output.call_args.args
        self.assertIsInstance(message, AbortReq)
        self.assertEqual(message.rid, canceled.rid)
        self.assertIs(owner, canceled)

    def test_abort_all_releases_each_queued_request(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        requests = [self._queued("one"), self._queued("two")]
        s.waiting_queue = requests.copy()
        with patch("sglang.srt.managers.scheduler.maybe_release_metadata_buffer") as release:
            s.abort_request(AbortReq(abort_all=True))
        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(release.call_count, 2)
        self.assertEqual(s.ipc_channels.send_to_tokenizer.send_output.call_count, 2)
        for req in requests:
            req.disagg_kv_sender.abort.assert_called_once()

    def test_decode_abort_preserves_kv_cleanup(self):
        s = self._scheduler(DisaggregationMode.DECODE)
        req = self._queued("cancel")
        s.waiting_queue = [req]
        recv = AbortReq(rid=req.rid)
        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            s.abort_request(recv)
        release.assert_called_once_with(req, s.tree_cache)
        s.disagg_decode_prealloc_queue.abort_matching.assert_called_once_with(recv)
        s.disagg_decode_transfer_queue.abort_matching.assert_called_once_with(recv)

    def test_chunked_abort_keeps_explicit_reason_and_clears_state(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        req = self._queued("chunked")
        reason = {"type": "abort", "status_code": 500, "message": "load failed"}
        req.to_finish = SimpleNamespace(to_json=lambda: reason)
        s.chunked_req = req
        s._pending_chunked_abort_req = req
        s.clear_pending_chunk_send = MagicMock()
        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release, patch(
            "sglang.srt.managers.scheduler.maybe_release_metadata_buffer"
        ):
            s.process_pending_chunked_abort()
        release.assert_called_once_with(req, s.tree_cache, is_insert=False)
        self.assertIsNone(s.chunked_req)
        self.assertIsNone(s._pending_chunked_abort_req)
        self.assertFalse(req.pending_bootstrap)
        message, owner = s.ipc_channels.send_to_tokenizer.send_output.call_args.args
        self.assertEqual(message.finished_reason, reason)
        self.assertIs(owner, req)


if __name__ == "__main__":
    unittest.main()
