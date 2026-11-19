"""Timing checks for the shared 60 Hz ForceGello replay buffer."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from ImplicitRDP.dataset.real_image_tactile_dataset import RealImageTactileDataset


class ForceGelloDatasetTest(unittest.TestCase):
    def test_future_joint_command_and_delayed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = zarr.open_group(str(Path(directory) / "replay_buffer.zarr"), mode="w")
            root.attrs["sample_hz"] = 60.0
            root.create_group("meta").array("episode_ends", np.array([120, 240]))
            data = root.create_group("data")
            n = 240
            joint_pos = np.zeros((n, 7), dtype=np.float32)
            joint_pos[:, 0] = np.arange(n)
            command = joint_pos.copy()
            command[:, 0] += 0.25
            command[:, 6] = 0.02
            image = np.broadcast_to(
                np.arange(n, dtype=np.uint8)[:, None, None, None],
                (n, 4, 4, 3),
            ).copy()
            for key, values in {
                "right_robot_command": command,
                "right_robot_command_valid": np.ones(n, dtype=bool),
                "right_robot_joint_pos": joint_pos,
                "right_robot_joint_vel": np.zeros((n, 7), dtype=np.float32),
                "right_robot_tcp_wrench": np.zeros((n, 6), dtype=np.float32),
                "external_img": image,
                "right_wrist_img": image,
            }.items():
                data.array(key, values)

            shape_meta = {
                "obs": {
                    "external_img": {"shape": [3, 4, 4], "type": "rgb", "hold_steps": 4},
                    "right_wrist_img": {"shape": [3, 4, 4], "type": "rgb", "hold_steps": 4},
                    "right_robot_joint_pos": {"shape": [7], "type": "low_dim"},
                    "right_robot_joint_vel": {"shape": [7], "type": "low_dim"},
                    "right_robot_tcp_wrench": {"shape": [6], "type": "low_dim"},
                },
                "extended_obs": {
                    "right_robot_joint_pos": {"shape": [7], "type": "low_dim"},
                    "right_robot_joint_vel": {"shape": [7], "type": "low_dim"},
                    "right_robot_tcp_wrench": {"shape": [6], "type": "low_dim"},
                },
                "action": {"key": "right_robot_command", "space": "joint", "shape": [7]},
            }
            dataset = RealImageTactileDataset(
                shape_meta, directory, horizon=20, n_obs_steps=5,
                pad_before=4, pad_after=15, sample_hz=20, sample_phase=0,
                visual_latency_steps=1, val_ratio=0.5)
            sample = dataset[10]
            self.assertEqual(sample["action"].shape, (20, 7))
            self.assertEqual(sample["obs"]["external_img"].shape, (5, 3, 4, 4))
            self.assertNotIn("right_robot_tcp_pose", sample["obs"])
            held_indices = dataset._observation_indices("external_img", 10, 5)
            self.assertLessEqual(len(np.unique(held_indices)), 2)
            current_source_index = int(sample["obs"]["right_robot_joint_pos"][-1, 0])
            self.assertAlmostEqual(float(sample["action"][4, 0]), current_source_index + 3.25)
            self.assertAlmostEqual(float(sample["action"][4, 6]), 0.020, places=6)
            expected_camera_source_index = ((current_source_index // 3 - 1) // 4) * 12
            self.assertAlmostEqual(
                float(sample["obs"]["external_img"][-1, 0, 0, 0]),
                expected_camera_source_index / 255.0,
                places=6,
            )
            source_ids = dataset.replay_buffer.source_episode_ids
            self.assertEqual(len(source_ids), 2)
            self.assertFalse(set(source_ids[dataset.train_mask]) & set(source_ids[dataset.val_mask]))
            normalizer = dataset.get_normalizer()
            self.assertEqual(normalizer["action"].params_dict["scale"].shape[0], 7)



if __name__ == "__main__":
    unittest.main()
