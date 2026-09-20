import unittest

from hardware.control import wait_for_change


class WaitForChangeTest(unittest.TestCase):
    def test_returns_when_readback_changes(self):
        values = iter([1.0, 1.0, 1.2])

        wait_for_change(
            lambda: next(values),
            1.0,
            threshold=0.1,
            timeout=0.1,
            poll_interval=0.0,
        )

    def test_times_out_when_readback_is_stuck(self):
        with self.assertRaises(TimeoutError):
            wait_for_change(
                lambda: 1.0,
                1.0,
                threshold=0.1,
                timeout=0.001,
                poll_interval=0.0,
            )


if __name__ == "__main__":
    unittest.main()
