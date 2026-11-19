# ForceGello ImplicitRDP training

Use the existing `ForceGello` conda environment. The offline training path uses
PyTorch, torchvision, NumPy, SciPy, Zarr, numcodecs, Hydra, OmegaConf,
Diffusers, Accelerate, timm, einops, OpenCV, wandb, dill, Numba,
threadpoolctl, loguru, and transforms3d. It does not import the ROS,
Flexiv, Open3D, or Pymunk runtime paths. Do not install this repository's
full legacy `requirements.txt` into `ForceGello` for this task.

From the `ImplicitRDP` repository root:

```bash
conda activate ForceGello
python -m scripts.check_forcegello_training_env
python train.py --config-name=train_forcegello_right_implicitrdp
```

The config reads `../i2rt/data/forcegello_zarr/replay_buffer.zarr`. The Zarr
must be complete; a `.partial` directory is not a training input. It reads
60 Hz recordings without rewriting them, taking phase 0 only to form one
20 Hz stream per episode. The 7D action is the absolute submitted follower
command at `t+50 ms`: six joint angles in radians and normalized gripper
opening. Joint position, joint velocity, and TCP wrench are sampled at `t`.
TCP position and orientation are not observations. Images are held at 5 Hz
and delayed one 20 Hz tick relative to proprioception. This is the single
stage ImplicitRDP Transformer, so it does not have RDP's separate slow
visual latent stage.

The task config uses the right arm, external camera, and right wrist camera.
Training defaults to CUDA device 0, batch size 8, and offline wandb logging.

After retraining with this config, verify the checkpoint without opening CAN
or cameras:

```bash
cd ../i2rt
python examples/implicitrdp_policy/run.py --checkpoint /path/to/checkpoint.ckpt --validate-only
```

Then run the same command without `--validate-only`. The yellow button toggles
policy control; Ctrl+C stops the robot. Online inference runs in a separate
thread so the 60 Hz motor loop continues. Each 20 Hz prediction targets the
submitted command for `t+50 ms`; the motor loop interpolates across three
60 Hz ticks and sets the six arm velocity targets to `(q_target-q_previous)/0.05`.
The gripper uses position only. A prediction arriving after its `t+50 ms`
deadline is discarded, and repeated misses cause a timeout instead of sending
an old action. Existing TCP-delta checkpoints are incompatible with this
7D joint-command config and must be retrained.
