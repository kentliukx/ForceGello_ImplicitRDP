"""Check the conda runtime without connecting to robot hardware."""

import argparse
import dataclasses
import importlib
import os
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)

    import numpy as np
    import cv2
    import torch
    import torchvision
    import torchaudio
    import zarr
    import rerun as rr
    import rclpy
    from cv_bridge import CvBridge
    from std_msgs.msg import String
    from ImplicitRDP.common.replay_buffer import ReplayBuffer

    print(f"NumPy: {np.__version__}; OpenCV: {cv2.__version__}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"TorchVision: {torchvision.__version__}; TorchAudio: {torchaudio.__version__}")
    print(f"ROS: {rclpy.__file__}", flush=True)
    if not os.path.realpath(rclpy.__file__).startswith(os.path.realpath(sys.prefix) + os.sep):
        raise RuntimeError("ROS is outside this Python environment; check PYTHONPATH.")

    for module in (
        "message_filters",
        "peft",
        "ImplicitRDP.model.vision.timm_obs_encoder",
        "ImplicitRDP.model.vision.transformer_obs_encoder",
        "ImplicitRDP.policy.diffusion_transformer_image_policy",
        "ImplicitRDP.policy.latent_diffusion_unet_image_policy",
        "ImplicitRDP.workspace.train_diffusion_transformer_image_workspace",
        "ImplicitRDP.dataset.real_image_tactile_dataset",
        "ImplicitRDP.real_world.ros_data_converter",
        "flexivrdk",
        "pyrealsense2",
    ):
        importlib.import_module(module)
    assert hasattr(rr, "Scalars") and hasattr(rr, "set_time")
    print("PASS: project, ROS, visualization and hardware SDK imports", flush=True)

    from ImplicitRDP.real_world.real_world_transforms import RealWorldTransforms

    first, second = RealWorldTransforms(), RealWorldTransforms()
    for field in dataclasses.fields(first):
        value = getattr(first, field.name)
        if isinstance(value, np.ndarray):
            assert not np.shares_memory(value, getattr(second, field.name))
    print("PASS: independent calibration defaults on Python 3.12", flush=True)

    replay = ReplayBuffer.create_empty_zarr()
    replay.add_episode({"action": np.zeros((3, 6), dtype=np.float32)})
    assert replay.n_steps == 3 and replay.n_episodes == 1
    print(f"PASS: Zarr {zarr.__version__} replay buffer write", flush=True)

    bridge = CvBridge()
    image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    restored = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    np.testing.assert_array_equal(restored, image[:, :, ::-1])
    print("PASS: ROS image conversion and color conversion", flush=True)

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    scheduler = DDIMScheduler(num_train_timesteps=10)
    scheduler.set_timesteps(2)
    sample = torch.zeros((1, 4, 6))
    result = scheduler.step(torch.zeros_like(sample), scheduler.timesteps[0], sample)
    assert torch.isfinite(result.prev_sample).all()
    print("PASS: diffusion scheduler step", flush=True)

    if torch.cuda.is_available():
        tensor = torch.ones((128, 128), device="cuda")
        product = tensor @ tensor
        torch.cuda.synchronize()
        assert product[0, 0].item() == 128
        print(f"PASS: GPU matrix multiplication on {torch.cuda.get_device_name(0)}", flush=True)
    elif args.require_cuda:
        raise RuntimeError("CUDA is unavailable; run nvidia-smi in the same terminal.")
    else:
        print("SKIP: CUDA unavailable (use --require-cuda to require a GPU)", flush=True)

    # A separate ROS domain and a unique topic keep this check away from normal topics.
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(domain_id=77)
    node = None
    try:
        node = rclpy.create_node("implicitrdp_environment_check")
        received = []
        topic = "/implicitrdp_environment_check_" + uuid.uuid4().hex
        subscription = node.create_subscription(String, topic, lambda msg: received.append(msg.data), 10)
        publisher = node.create_publisher(String, topic, 10)
        deadline = time.monotonic() + 10
        while not received and time.monotonic() < deadline:
            publisher.publish(String(data="implicitrdp_ok"))
            rclpy.spin_once(node, timeout_sec=0.1)
        if received != ["implicitrdp_ok"]:
            raise RuntimeError(f"ROS publish/subscribe check failed: {received}")
        node.destroy_subscription(subscription)
        print("PASS: ROS 2 publish/subscribe", flush=True)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

    print("Environment checks passed. Robot and camera hardware have not been tested.")


if __name__ == "__main__":
    main()
