#!/usr/bin/env python3
import pytest
import unittest
import time
import numpy as np
from dataclasses import dataclass
from tabulate import tabulate
from typing import List

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.mock import mock_messages
from openpilot.selfdrive.car.car_helpers import write_car_param
from openpilot.system.hardware.tici.power_monitor import get_power
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.selfdrive.manager.manager import manager_cleanup

SAMPLE_TIME = 8        # seconds to sample power
MAX_WARMUP_TIME = 10   # max amount of time to wait for process to warmup
WARMUP_TIME = 4        # need 4 seconds worth of messages for warmup to complete

@dataclass
class Proc:
  name: str
  power: float
  msgs: List[str]
  rtol: float = 0.05
  atol: float = 0.12

PROCS = [
  Proc('camerad', 2.1, msgs=['roadCameraState', 'wideRoadCameraState', 'driverCameraState']),
  Proc('modeld', 1.12, atol=0.2, msgs=['modelV2']),
  Proc('dmonitoringmodeld', 0.4, msgs=['driverStateV2']),
  Proc('encoderd', 0.23, msgs=[]),
  Proc('mapsd', 0.05, msgs=['mapRenderState']),
  Proc('navmodeld', 0.05, msgs=['navModel']),
]


@pytest.mark.tici
class TestPowerDraw(unittest.TestCase):

  def setUp(self):
    write_car_param()

    # wait a bit for power save to disable
    time.sleep(5)

  def tearDown(self):
    manager_cleanup()

  def measure_msg_count_and_power(self, proc, max_time, msgs_expected):
    socks = {msg: messaging.sub_sock(msg) for msg in proc.msgs}

    msg_count = 0
    start_time = time.time()
    powers = []

    while True:
      powers.append(get_power(1))
      for sock in socks.values():
        msg_count += len(messaging.drain_sock_raw(sock))
      if msg_count > msgs_expected or (time.time() - start_time) > max_time:
        break

    return msg_count, np.mean(powers)

  def get_expected_msg_count(self, proc, time):
    return int(sum(time * SERVICE_LIST[msg].frequency for msg in proc.msgs))

  @mock_messages(['liveLocationKalman'])
  def test_camera_procs(self):
    baseline = get_power()

    prev = baseline
    used = {}
    msg_counts = {}

    for proc in PROCS:
      managed_processes[proc.name].start()

      msgs_expected = self.get_expected_msg_count(proc, WARMUP_TIME)
      msgs_received, _ = self.measure_msg_count_and_power(proc, MAX_WARMUP_TIME, msgs_expected)

      with self.subTest(msg=f'warmup failed for {proc}'):
        np.testing.assert_allclose(msgs_expected, msgs_received, rtol=.02, atol=2)

      msgs_expected = self.get_expected_msg_count(proc, SAMPLE_TIME)
      msgs_received, now = self.measure_msg_count_and_power(proc, SAMPLE_TIME, msgs_expected)

      with self.subTest(msg=f'msg count failed for {proc}'):
        np.testing.assert_allclose(msgs_expected, msgs_received, rtol=.02, atol=2)

      msg_counts[proc.name] = msgs_received
      used[proc.name] = now - prev
      prev = now

      with self.subTest(msg=f'power failed for {proc}'):
        np.testing.assert_allclose(used[proc.name], proc.power, rtol=proc.rtol, atol=proc.atol)

    manager_cleanup()

    tab = [['process', 'expected (W)', 'measured (W)', '# msgs expected', '# msgs received']]
    for proc in PROCS:
      tab.append([proc.name, round(proc.power, 2), round(used[proc.name], 2), msgs_expected, msgs_received])
    print(tabulate(tab))
    print(f"Baseline {baseline:.2f}W\n")


if __name__ == "__main__":
  pytest.main()
