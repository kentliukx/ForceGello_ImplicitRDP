# YAM 接入 ImplicitRDP：采集数据与驱动接口约定

本文根据本仓库当前的采集、转换、后处理和训练配置整理。
“现有实现”描述已经存在的代码；“建议接口”是待接入 YAM 时采用的约定，不代表驱动已经实现。
目前尚未取得实验室修改后的 I2RT 代码，也没有确认相机和力传感器型号。

## 1. 先明确采集内容

每段示范是一个 episode：从一次任务开始到结束的一段连续时间序列。
当前视觉力策略的核心观测是彩色图像、实际末端位姿、实际末端外力/力矩。
还需要时间戳和 episode 边界。任务需要开合夹爪时，还应记录夹爪状态和示教目标。

机器人状态与图像可以由不同程序采集；不必把相机读取和文件写入塞进 YAM 控制循环。
驱动提供状态，桥接程序发布 ROS 消息，采集器负责同步和存储。

## 2. 建议 YAM 状态接口

建议新增适配层的状态快照接口，字段如下。这不是现有 I2RT API 的固定字段名。

| 字段 | 形状/类型 | 单位与语义 |
|---|---|---|
| `timestamp_ns` | `int64` | 状态采样时刻，统一时基的纳秒时间戳 |
| `tcp_pose` | 浮点 `[7]` | `[x,y,z,qw,qx,qy,qz]`，位置为 m，姿态为单位四元数，表示 TCP 相对机械臂基座的姿态 |
| `tcp_twist` | 浮点 `[6]` | `[vx,vy,vz,wx,wy,wz]`，m/s 和 rad/s；建议表示 TCP 点相对基座的速度，并用基座坐标轴表达 |
| `tcp_wrench` | 浮点 `[6]` | `[Fx,Fy,Fz,Tx,Ty,Tz]`，N 和 N·m；为兼容原有后处理，在 TCP 原点、TCP 坐标轴下表达外力/力矩 |
| `gripper_width` | 浮点标量 | 实际夹爪开口宽度，m；不是未经标定的电机角度或 0–1 开合比例 |
| `gripper_force` | 浮点标量或缺失 | 夹爪力，N；缺失时明确标记，适配层应处理缺失，不能标成真实测量 |
| `joint_pos` | 浮点 `[n]`，建议附带 | 实测机械臂关节角，rad，明确关节顺序，与夹爪分开 |
| `joint_vel` | 浮点 `[n]`，建议附带 | 实测关节角速度，rad/s |
| `joint_torque` | 浮点 `[n]`，有则附带 | 关节力矩，N·m，区分实测、估计、控制命令 |

若当前接口只能提供关节角，适配层可通过准确的机器人模型、关节顺序和 TCP 工具偏移计算末端位姿。
速度也可通过关节速度和雅可比计算。TCP 必须是实际操作点，不应把法兰原点误当成夹爪接触点。

`tcp_wrench` 需要明确来源：独立六维传感器，或经过验证的外力估计。
电机电流、关节总力矩、夹爪握力与 TCP 六维外力不是同一种数据。
力矩估计还涉及机器人动力学、重力/摩擦等补偿和雅可比；不能仅把六个关节力矩改名为六维 wrench。
没有 wrench 时，先保存真实的关节反馈和已有观测，并明确缺失；不要填全零后当作力策略数据训练。

需要记录力的符号约定、传感器零偏、工具质量/重心、补偿方法和传感器到 TCP 的标定变换。
变换 wrench 时既要考虑坐标轴旋转，也要考虑测力原点偏移对力矩的影响。
本仓库原 Flexiv server 读取的是 `ext_wrench_in_tcp`，虚拟目标后处理会用 TCP 姿态把局部力偏移转到基座坐标。

## 3. ROS 桥接层对应关系

暂按单臂复用现有配置中的 `right` 命名；这只是字段槽位，不要求你的实体机器人一定装在右侧。

| 话题 | 消息类型 | 内容 |
|---|---|---|
| `/right_tcp_pose` | `geometry_msgs/msg/PoseStamped` | 实际 TCP 位姿；建议 `header.frame_id` 为真实基座坐标系 |
| `/right_tcp_vel` | `geometry_msgs/msg/TwistStamped` | 实际 TCP 线速度与角速度；写明表达坐标系 |
| `/right_tcp_wrench` | `geometry_msgs/msg/WrenchStamped` | TCP 坐标系表达、TCP 原点的外力和力矩；填写对应 TCP frame |
| `/right_gripper_state` | `sensor_msgs/msg/JointState` | 原转换器读取 `position[0]` 为宽度、`effort[0]` 为夹爪力，这是本项目的使用约定 |
| `/right_wrist_camera/color/image_raw` | 图像消息 | 相机若安装在手腕，可复用此名称；外部相机应使用对应外部相机字段并修改任务配置 |

所有消息需要有效的 `header.stamp`。同一状态快照的几条机器人消息可用同一个采样时间戳。
图像应使用图像获取时间；不同设备的时钟需对齐，不能直接混用设备启动后的相对时间和主机 epoch 时间。
同一主机下也需要核对时间戳来源。原转换器不根据 `frame_id` 自动执行 TF 变换，必须在进入转换器前统一数据语义。

I2RT 的公开 `Robot` 协议已有读取关节位置、状态、观测以及发送关节命令等接口；
可以保留底层接口，通过额外适配层对接本表，具体仍以你的修改版本为准。
[官方 Robot 协议源码](https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/robots/robot.py)

### 单条消息的字段

- `PoseStamped`：`pose.position.x/y/z` 为 m，`pose.orientation.x/y/z/w` 按四元数字段名赋值。
  ROS 字段的书写顺序不改变适配层数组约定；不要误把 `[qw,qx,qy,qz]` 当成 `[qx,qy,qz,qw]`。
- `TwistStamped`：`twist.linear.x/y/z` 为 m/s，`twist.angular.x/y/z` 为 rad/s。
- `WrenchStamped`：`wrench.force.x/y/z` 为 N，`wrench.torque.x/y/z` 为 N·m。
- 本项目的夹爪 `JointState`：`name=['right_gripper']`、`position=[width_m]`、`effort=[force_n]`。
  没有力测量时必须相应修改转换器的缺失字段处理。
- 标准 raw `Image`：通过 `CvBridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')` 构造，
  设置采样时间和相机 optical frame；这要求把原 JPEG 解码路径改成配套的 CvBridge 解码。

### 轨迹开始与结束

现有实现没有开始/停止录制的话题或服务。启动 `record_data.py` 后开始接收和缓存同步数据，
按 `Ctrl+C` 退出时，只有传入了 `--save_to_disk` 才会保存整个缓存。
示教控制器的 `Pause` 键用于切换/控制拖动示教状态，不是数据集 episode 的边界信号。

建议 YAM 接入时由采集器新增以下服务，底层状态和相机话题始终发布：

| 建议新增的服务 | 类型 | 行为 |
|---|---|---|
| `/recorder/start` | `std_srvs/srv/Trigger` | 空闲时新建 episode，记录开始时间，清空上次片段状态；重复开始返回失败 |
| `/recorder/stop` | `std_srvs/srv/Trigger` | 记录结束时间，停止向该片段添加样本并保存；返回保存结果及文件位置 |
| `/recorder/discard` | `std_srvs/srv/Trigger` | 放弃当前片段，返回空闲状态 |

以上是接口设计，当前仓库还没有实现这些服务。
开始/结束边界按与消息相同的时基定义，按采样时间保留片段内的数据，防止同步队列中的旧帧进入新片段。
接触式任务可以在稳定初始状态时开始，在任务完成后停止，复位动作在录制之外进行。
停止录制不等于停止机械臂运动，也不代替机器人自己的停止控制。
每段建议另外记录 `episode_id`、起止时间、任务名和成功/失败结果。

## 4. 图像格式与采样频率

图像数组为 `uint8[H,W,3]`，通道顺序必须固定并记录。示例后处理输出大小是 `640×360`，
Zarr 内存储为 `[T,360,640,3]`；训练 dataset 再转为 `[T,3,360,640]` 并除以 255。
不必让驱动提前输出训练用的 CHW 浮点图像。

注意：现有 USB publisher 用 `cv2.imencode('.jpg', ...)` 压缩，然后把 JPEG 字节放进
`sensor_msgs/Image.data`，转换器对应使用 `cv2.imdecode`；普通相机路径解码后为 BGR。
这与标准 ROS `Image` 的未压缩像素布局不同。接新相机时应选择一种完整配套的方案：

- 新接口建议使用标准 `Image`（`rgb8`/`bgr8`，通过 CvBridge 解码），或标准 `CompressedImage`；相应修改转换器和订阅消息类型。
- 如果暂时沿用旧 publisher 和旧转换器，保持其 JPEG 传输与解码方式一致。

不要只换相机 publisher、保留旧解码器，也不要把 BGR 图像在没有转换的情况下声明为 RGB。
[ROS Image 消息定义](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/Image.msg)

示例 `real_flip_one_usb_camera_kineteach_10fps.yaml` 为相机 10 Hz、机器人发布 20 Hz。
这是该任务的配置，不是 ImplicitRDP 所有任务必须使用的频率。
建议保留硬件实际产生的机器人/力高频数据和各自时间戳，后处理时再重采样到训练频率。
当前采集器把各话题近似同步到同一帧再保存，不会自动保留图像帧之间的所有高频力样本；
若需要高频力历史，应额外保存原始流并扩展后处理，不能靠重复低频样本获得。

原采集器同步参数为 `queue_size=25, slop=0.1`，即允许最多约 100 ms 的时间戳差。
这只是当前同步器的容差，不代表接触任务可接受 100 ms 的观测误差。
接入时应测量实际图像与状态偏差，根据设备频率、延迟和任务要求调整。

时间建议保存为 `int64` 纳秒或 `float64` 秒。当前后处理把 Zarr `timestamp` 写为 `float32`；
如果输入是 epoch 秒，会丢失帧间时间精度。正式接入时应改为 `float64` 秒或保留 `int64` 纳秒。

## 5. 原始文件与训练文件

驱动不需要直接生成 Zarr，也不需要输出 19 维训练动作。
当前流程是：

```text
机器人状态 + 相机消息
          ↓ 同步、转换
每段示范一个 .pkl
          ↓ post_process_data.py
replay_buffer.zarr
```

原始 pickle 的结构为：

```python
{
    'sensor_message_list': SensorMessageList(
        sensorMessages=[SensorMessage(...), SensorMessage(...), ...]
    )
}
```

其中单帧的相关字段为：

| 原始 `SensorMessage` 字段 | 形状 | 说明 |
|---|---|---|
| `timestamp` | float | 同步组中最大的消息时间戳，秒；原实现不保留每个话题的原始时间戳 |
| `rightRobotTCP` | `[6]` | `[x,y,z,roll,pitch,yaw]`，m、rad；ROS 四元数经 `transforms3d` 默认 `sxyz` 约定转换 |
| `rightRobotTCPVel` | `[6]` | `[vx,vy,vz,wx,wy,wz]` |
| `rightRobotTCPWrench` | `[6]` | `[Fx,Fy,Fz,Tx,Ty,Tz]` |
| `rightRobotGripperState` | `[2]` | `[width,force]` |
| `rightWristCameraRGB` | `[H,W,3]` | 原普通相机实现解码为 BGR，字段名不保证数据实际为 RGB |

注意三个不同层次的姿态格式：驱动适配层建议使用 7 维位置+四元数；
原始 SensorMessage 使用 6 维位置+RPY；训练 Zarr 使用 9 维位置+连续 6D 旋转表示。
最后一种旋转表示是旋转矩阵前两列依次拼接，不是 6 个欧拉角。

单臂视觉力任务相关的训练数据子集为：

```text
replay_buffer.zarr/
  data/
    timestamp                 [T]        # 建议 float64 秒
    right_wrist_img           [T,H,W,3]  # uint8
    right_robot_tcp_pose      [T,9]      # float32
    right_robot_tcp_vel       [T,6]      # float32
    right_robot_tcp_wrench    [T,6]      # float32
    right_robot_gripper_width [T,1]      # float32
    right_robot_gripper_force [T,1]      # float32
    target                    [T,D]      # float32，当前时刻构造的标签状态
    action                    [T,D]      # float32，由后处理生成
  meta/
    episode_ends              [E]        # int64，累计终点/右开区间索引
```

这是相关字段子集，不是声称当前脚本只会写这些字段；原脚本仍会输出其他相机、左臂及默认占位数据。
每段示范分别保存。原脚本按数字文件名排序，例如 `1.pkl`、`2.pkl`，并记录每段的累计结束位置。
建议额外保存每段的任务名称、成功/失败、操作者、标定版本、相机参数和驱动版本等 metadata。

## 6. action 到底是什么

当前采集器没有记录真实发送的控制命令，代码中保留了 `TODO: record actual action`。
后处理先由实测状态构造 `target/state`，再用下一帧的标签状态作为当前帧的 `action`，并单独处理 episode 末帧。

以 `right_arm_6DOF_virtual_target_stiffness` 为例：

- 9 维实际末端位姿。
- 根据局部受力与启发式刚度构造的 9 维虚拟目标位姿。
- 1 维启发式刚度，共 19 维 Zarr 标签。

模型配置还会将旋转表示转换为 RPY，形成 13 维模型动作。这些都是后处理/模型侧的表示。
这里的虚拟目标和刚度不是从示教控制器直接读取的真实命令。
原脚本中的力阈值、刚度范围和力的符号不能未经验证直接照搬到 YAM 控制器。

建议修改 YAM 时同时把实际发送的控制命令及其时间戳记录下来，例如 `q_cmd`、夹爪目标，
或者真正使用的 `tcp_target`、刚度/阻尼参数，并注明控制模式。它们与实测状态分别保存。
原流程不会自动使用这些新增字段，需要后续明确训练动作定义、延迟与标签对齐方式。
示例 flip 配置的右臂动作不包含夹爪目标；如果任务需要策略控制夹爪，还要扩展 action、shape_meta 和执行接口。

### 使用主臂/从臂遥操作时

主臂是人的示教输入，从臂是实际执行任务的机器人。针对自主部署，推荐记录：

- `obs_t`：从臂当前实测状态、相机图像、从臂末端受力及必要的历史观测。
- `action_t`：同一控制时刻真正发给从臂的目标命令。主臂读数经过标定映射、限幅、滤波等处理后才形成这个命令。
- 附加原始数据：主臂读数、从臂实测状态、发送命令、各自时间戳，以及控制模式/增益，分别保存。

例：主臂某关节角为 0.80 rad，映射/限幅后的从臂目标为 0.75 rad，从臂当前实测为 0.70 rad。
以关节目标作为 action 时，标签是 0.75，而当前关节观测是 0.70；主臂 0.80 只是命令来源。
原项目的“下一帧实测状态标签”是另一种标签构造方法，不应与真实下发命令混淆。

部署时策略替代主臂的命令来源。若 action 是关节目标，可对接 YAM 关节位置控制；
若 action 是末端目标，需要相应的 IK/笛卡尔控制接口，训练和部署应保持相同语义。
若从关节命令生成末端目标标签，应对映射后的从臂目标关节量，使用从臂模型和相同 TCP 定义进行正运动学转换。
改为关节 action 会涉及模型维度、归一化、预处理和执行器，不是只改字段名。

原仓库的 `left/right` 表示两只执行臂，不自动表示遥操作的主臂/从臂。
主从各一只的系统，可能仍然只有一只实际执行臂；不要把主臂误当成另一个任务执行臂的观测槽位。

另需注意：当前 `RealRobotEnvironment.execute_action()` 对
`right_arm_6DOF_virtual_target_stiffness` 明确抛出 `NotImplementedError`，runner 也没有完整的该类型执行分支。
19 维训练标签不能直接视为已支持的 YAM 部署命令。
原 Flexiv 控制封装调用厂商的笛卡尔运动/阻抗接口；YAM 的控制方案应以实际修改后的驱动为准。

## 7. 接入前需要修正的现有假设

1. `device_mapping_utils.py` 固定订阅左右两臂共 8 条状态话题。
   单 YAM 接入应让话题清单和转换器只读取实际使用的臂；否则同步器会一直等未发布的话题。
   无须为了通过同步器而伪造另一台机器人的测量数据。
2. 同文件对每个 USB 相机额外订阅 marker 信息；原 USB publisher 会发送 dummy marker。
   普通相机接入应删除不需要的 marker 订阅及相应转换假设，否则也会阻塞同步。
3. 转换器固定读取夹爪 `effort[0]`。若 YAM 没有夹爪力测量，应明确支持该字段缺失，
   避免空数组越界，也避免把占位值误当成测量。
4. 相机消息编码、坐标系、四元数顺序和时间戳精度按上文统一。
5. `record_data.py`、`post_process_data.py` 和训练脚本保留了原作者的数据路径、默认任务和传感器模式。
   正式采集前要改为 YAM 的路径和真实设备配置，不能直接按原默认多相机/双臂配置采集。

## 8. 对照源码

- [字段定义](../ImplicitRDP/common/data_models.py)
- [ROS 到原始记录的转换](../ImplicitRDP/real_world/ros_data_converter.py)
- [机器人状态发布](../ImplicitRDP/real_world/publisher/bimanual_robot_publisher.py)
- [采集与保存](../ImplicitRDP/real_world/teleoperation/data_recorder.py)
- [订阅话题清单](../ImplicitRDP/real_world/device_mapping/device_mapping_utils.py)
- [动作生成与 Zarr 写入](../post_process_data.py)
- [示例任务的观测和动作维度](../ImplicitRDP/config/task/real_flip_image_wrench_implicitrdp_10fps.yaml)
