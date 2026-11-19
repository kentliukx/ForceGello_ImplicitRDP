"""Check only the packages used by the offline ForceGello training path."""

import importlib
import sys


MODULES = (
    "torch", "torchvision", "numpy", "scipy", "zarr", "numcodecs",
    "hydra", "omegaconf", "diffusers", "accelerate", "timm",
    "einops", "cv2", "wandb", "dill", "numba", "threadpoolctl",
    "loguru", "transforms3d",
)


def main():
    failed = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            failed.append((name, exc))
    if failed:
        for name, exc in failed:
            print(f"{name}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    from ImplicitRDP.dataset.real_image_tactile_dataset import RealImageTactileDataset
    from ImplicitRDP.workspace.train_diffusion_transformer_image_workspace import (
        TrainDiffusionTransformerImageWorkspace,
    )
    assert RealImageTactileDataset and TrainDiffusionTransformerImageWorkspace

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to this process")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; bf16: {torch.cuda.is_bf16_supported()}")
    print("ForceGello ImplicitRDP offline training environment is ready")


if __name__ == "__main__":
    main()
