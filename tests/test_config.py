import os
import unittest
from unittest.mock import patch

from rainbow.config import ConfigurationError, HardwareConfig


class HardwareConfigTest(unittest.TestCase):
    def test_requires_explicit_hardware_endpoints(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationError, "DLC_HOST"):
                HardwareConfig.from_values()

    def test_reads_settings_from_environment(self):
        environment = {
            "DLC_HOST": "laser-controller.example",
            "WAVEMETER_URL": "https://wavemeter.example/",
            "WAVEMETER_PORT": "8443",
            "OSCILLOSCOPE_RESOURCE": "USB",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = HardwareConfig.from_values()

        self.assertEqual(config.dlc_host, "laser-controller.example")
        self.assertEqual(config.wavemeter_url, "https://wavemeter.example")
        self.assertEqual(config.wavemeter_port, 8443)
        self.assertEqual(config.oscilloscope_resource, "USB")

    def test_explicit_values_override_environment(self):
        environment = {
            "DLC_HOST": "environment-controller.example",
            "WAVEMETER_URL": "https://environment-wavemeter.example",
            "WAVEMETER_PORT": "8000",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = HardwareConfig.from_values(
                dlc_host="cli-controller.example",
                wavemeter_url="https://cli-wavemeter.example",
                wavemeter_port=9000,
            )

        self.assertEqual(config.dlc_host, "cli-controller.example")
        self.assertEqual(config.wavemeter_url, "https://cli-wavemeter.example")
        self.assertEqual(config.wavemeter_port, 9000)

    def test_rejects_invalid_url_and_port(self):
        with self.assertRaisesRegex(ConfigurationError, "absolute http"):
            HardwareConfig.from_values(
                dlc_host="controller.example",
                wavemeter_url="wavemeter.example",
                wavemeter_port=8000,
            )
        with self.assertRaisesRegex(ConfigurationError, "between 1 and 65535"):
            HardwareConfig.from_values(
                dlc_host="controller.example",
                wavemeter_url="https://wavemeter.example",
                wavemeter_port=70000,
            )


if __name__ == "__main__":
    unittest.main()
