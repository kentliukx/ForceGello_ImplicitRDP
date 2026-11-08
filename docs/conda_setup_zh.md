# Ubuntu 24.04 的 conda 环境配置

本方案使用 RoboStack，在同一个 conda 环境内安装 ROS 2 Jazzy 和 Python。
原项目使用 Ubuntu 22.04 + Humble + venv；这里提供的是本机适配方案。
环境检查通过不代表已经完成真实机械臂的部署验证。

本机已验证的主要版本：Python 3.12.16、ROS 2 Jazzy、PyTorch 2.2.2+cu121、
NumPy 1.26.4、OpenCV 4.13.0、Zarr 2.18.7、Diffusers 0.33.1。
RTX 4070 Laptop GPU 的 CUDA 运算、ROS 发布/订阅、图像颜色转换、项目模块导入、
ReplayBuffer 写入和训练配置解析均已通过。

## 1. 日常进入环境

在新终端运行这一行即可：

```bash
conda activate implicitrdp
```

环境位于 conda 默认目录 `~/miniconda3/envs/implicitrdp`，可按名称管理。
每个运行项目程序的终端都要激活它。退出时运行 `conda deactivate`。
只有终端找不到 `conda` 命令时，才需要先运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

不要在这个环境中执行 `source /opt/ros/jazzy/setup.bash`。
RoboStack 在 `conda activate` 时自动设置 ROS 环境。
如果 `.bashrc` 中有自动加载系统 ROS 的命令，请在使用 RoboStack 的终端中禁用该行后重新打开终端。

确认当前 Python 和 ROS 来自项目环境：

```bash
which python
python -c "import rclpy; print(rclpy.__file__)"
```

两者的路径都应该包含 `miniconda3/envs/implicitrdp`。

## 2. 检查环境

激活后，在项目根目录运行：

```bash
cd ~/Forcegello/Forcegello_ImplicitRDP
python -m pip check
python check_environment.py --require-cuda
```

检查内容包括项目模块导入、Zarr 数据写入、ROS 图像转换、扩散调度器、
GPU 矩阵运算，以及独立测试话题的 ROS 发布/订阅。
脚本不连接机械臂、不打开相机，也不启动机器人控制服务。

查看显卡：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

PyTorch 使用项目 README 指定的 `2.2.2+cu121`。
`nvidia-smi` 显示的 CUDA 版本表示驱动支持的上限，不必与 PyTorch 的 CUDA 12.1 完全一致。

## 3. 从头创建环境

这部分供复现使用；已经安装好的环境不需要重复创建。
先进入项目目录，加载 conda，然后运行：

```bash
CONDA_CHANNEL_PRIORITY=strict conda env create \
  --file environment.yml

conda activate implicitrdp

python -m pip install \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  -r requirements.txt -c constraints-conda.txt 'setuptools<81'

python -m pip check
python check_environment.py --require-cuda
```

- `environment.yml`：ROS、Python 3.12、NumPy 1.26.4 和 OpenCV 的 conda 依赖。
- `requirements.txt`：项目原有的 Python 依赖。
- `constraints-conda.txt`：为本方案固定关键版本，避免安装与旧版 PyTorch 或项目接口不兼容的新版本。
  例如，项目调用了 `zarr.MemoryStore()`，需要使用 Zarr 2。
- `RealWorldTransforms` 的 NumPy 默认值已改为 `default_factory`，适配 Python 3.11 及以上的 dataclass 检查，并避免实例之间共享可变数组。
- OpenCV 由 conda 提供，保留与 `cv_bridge` 配套的本机库。

安装完成后，不要直接运行 `pip install --upgrade -r requirements.txt` 或 `conda update --all`。
需要变更依赖时保留约束文件，并重新运行上面的环境检查。

## 4. 环境就绪后的下一步

先确认机械臂、力传感器和相机型号，再按 `customized_deployment_guide.md` 配置硬件。
项目默认使用 Flexiv 机械臂；安装 `flexivrdk` 并不意味着其他品牌机械臂可以直接运行。

训练脚本还需要实际数据集路径；`train_implicitrdp.sh` 中当前保留的是原作者的绝对路径。
本机为 8 GB 显存，训练时需要根据模型和输入大小调整 batch size。

官方参考：[RoboStack conda 安装文档](https://robostack.github.io/conda.html)。
