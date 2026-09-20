import importlib.util
from types import SimpleNamespace
import unittest

from rainbow.constants import IMAGE_SIZE, SENSOR_DIM


def memory_args(torch, image_only):
    return SimpleNamespace(
        device=torch.device("cpu"),
        history_length=2,
        discount=0.99,
        multi_step=2,
        priority_weight=0.4,
        priority_exponent=0.5,
        image_only=image_only,
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ReplayMemorySchemaTest(unittest.TestCase):
    def test_image_only_transition(self):
        import torch

        from rainbow.memory import ReplayMemory

        memory = ReplayMemory(memory_args(torch, image_only=True), capacity=8)
        state = {"image": torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE)}

        memory.append(state, action=1, reward=0.5, terminal=False)

        self.assertNotIn("sensor", memory.transitions.data.dtype.names)
        self.assertEqual(memory.transitions.data[0]["action"], 1)

    def test_multimodal_transition_preserves_sensor_values(self):
        import torch

        from rainbow.memory import ReplayMemory

        memory = ReplayMemory(memory_args(torch, image_only=False), capacity=8)
        sensor = torch.arange(SENSOR_DIM, dtype=torch.float32)
        state = {
            "image": torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE),
            "sensor": torch.stack([torch.zeros(SENSOR_DIM), sensor]),
        }

        memory.append(state, action=2, reward=-0.5, terminal=True)

        self.assertIn("sensor", memory.transitions.data.dtype.names)
        self.assertTrue(
            torch.equal(torch.from_numpy(memory.transitions.data[0]["sensor"]), sensor)
        )


if __name__ == "__main__":
    unittest.main()
