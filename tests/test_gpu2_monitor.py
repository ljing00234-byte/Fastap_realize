import unittest
from unittest.mock import patch

from scripts import gpu2_monitor as monitor


class ProcessOwnershipTests(unittest.TestCase):
    def test_fastap_reserves_its_declared_peak_until_cuda_has_allocated_it(self):
        script = str(monitor.ROOT / "scripts/train_fastap.py").encode()
        cmdline = b"\0".join([b"python", script, b"--worker", b"--expected-peak-mib", b"6000"])
        with patch.object(monitor.Path, "read_bytes", return_value=cmdline):
            self.assertEqual(monitor.pending_allocations({41: 1024}), 4976)

    def test_exited_pid_remains_in_first_device_query(self):
        state = {"compute_processes": {
            41: {"executable": "[Not Found]", "used_mib": 200},
            42: {"executable": "python", "used_mib": 300},
        }}
        with patch.object(monitor.Path, "stat", side_effect=FileNotFoundError), \
             patch.object(monitor, "query", side_effect=[[["41"], ["42"]], [["42"]]]) as query, \
             patch.object(monitor.time, "sleep"):
            self.assertEqual(monitor.project_processes(state, own_pid=42), {42: 300})
        self.assertEqual(query.call_count, 2)

    def test_persistent_unaccounted_pid_blocks_monitor(self):
        with patch.object(monitor.Path, "stat", side_effect=FileNotFoundError), \
             patch.object(monitor, "query", return_value=[["41"]]) as query, \
             patch.object(monitor.time, "monotonic", side_effect=[1, 1, 2, 3]), \
             patch.object(monitor.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "ownership of GPU process 41"):
                monitor.belongs_to_project(41, "[Not Found]")
        self.assertEqual([call.kwargs["timeout"] for call in query.call_args_list], [2, 1])

    def test_unreadable_process_blocks_immediately(self):
        with patch.object(monitor.Path, "stat", side_effect=PermissionError), \
             patch.object(monitor, "query") as query:
            with self.assertRaisesRegex(RuntimeError, "ownership of GPU process 41"):
                monitor.belongs_to_project(41, "python")
        query.assert_not_called()
