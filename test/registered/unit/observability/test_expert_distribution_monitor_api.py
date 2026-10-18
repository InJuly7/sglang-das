"""Control-plane/API contract: one arm for concurrent requests, status fanout."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin


class FakeManager(TokenizerControlMixin):
    def auto_create_handle_loop(self):
        pass


class TestMonitorAPI(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_share_one_arm_and_wait_for_ack(self):
        manager = FakeManager()
        gate = asyncio.Event()
        calls = []

        async def communicate(req):
            calls.append(req)
            await gate.wait()
            return [SimpleNamespace(status={"state": "RECORDING"})]

        manager.expert_distribution_monitor_communicator = communicate
        with patch(
            "sglang.srt.observability.expert_distribution_monitor.MonitorConfig.from_env",
            return_value=SimpleNamespace(enabled=True),
        ):
            tasks = [
                asyncio.create_task(manager.maybe_arm_expert_distribution_monitor())
                for _ in range(8)
            ]
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(not t.done() for t in tasks))
            gate.set()
            await asyncio.gather(*tasks)
            self.assertEqual(calls[0].action, "arm")
            self.assertEqual(len(calls[0].run_id), 32)

    async def test_disabled_sends_nothing(self):
        manager = FakeManager()
        manager.expert_distribution_monitor_communicator = AsyncMock()
        with patch(
            "sglang.srt.observability.expert_distribution_monitor.MonitorConfig.from_env",
            return_value=SimpleNamespace(enabled=False),
        ):
            await manager.maybe_arm_expert_distribution_monitor()
        manager.expert_distribution_monitor_communicator.assert_not_called()

    async def test_status_fanout_uses_separate_request(self):
        manager = FakeManager()
        manager.expert_distribution_monitor_communicator = AsyncMock(
            return_value=[
                SimpleNamespace(status={"state": "DONE", "rank": 0}),
                SimpleNamespace(status={"state": "DONE", "rank": 1}),
            ]
        )
        result = await manager.expert_distribution_monitor_status()
        self.assertEqual(len(result["ranks"]), 2)
        self.assertEqual(
            manager.expert_distribution_monitor_communicator.call_args.args[0].action,
            "status",
        )

    async def test_http_status_endpoint(self):
        manager = SimpleNamespace(
            expert_distribution_monitor_status=AsyncMock(
                return_value={"ranks": [{"state": "ARMED", "rank": 0}]}
            )
        )
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
        ):
            # No lifespan: this endpoint does not launch a model in a unit test.
            response = TestClient(http_server.app).get("/expert_distribution_monitor")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ranks"][0]["state"], "ARMED")


if __name__ == "__main__":
    unittest.main()
