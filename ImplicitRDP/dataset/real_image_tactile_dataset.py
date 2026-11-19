from typing import Dict, List
import torch
import numpy as np
import os
import time
from threadpoolctl import threadpool_limits
import copy
import tqdm
import pickle
from ImplicitRDP.common.pytorch_util import dict_apply
from ImplicitRDP.dataset.base_dataset import BaseImageDataset
from ImplicitRDP.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ImplicitRDP.common.replay_buffer import ReplayBuffer
from ImplicitRDP.common.sampler import (
    SequenceSampler, get_val_mask, get_val_mask_one_by_one, downsample_mask)
from ImplicitRDP.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)
from ImplicitRDP.common.action_utils import (
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions,
    matrix_actions_to_rpy_actions
)
from ImplicitRDP.common.space_utils import pose_3d_9d_to_homo_matrix_batch
from ImplicitRDP.real_world.real_world_transforms import RealWorldTransforms

from loguru import logger

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=True,
                 rpy_for_rotation=False,
                 transform_params=None,
                 sample_hz=None,
                 sample_phase=None,
                 visual_latency_steps=0,
                 ):
        assert os.path.isdir(dataset_path)

        self.action_key = shape_meta['action'].get('key', 'action')
        self.action_space = shape_meta['action'].get('space', 'cartesian')
        self.forcegello_sampling = sample_hz is not None
        if sample_hz is not None and (self.action_space not in ('joint', 'tcp_delta') or delta_action or relative_action or rpy_for_rotation):
            raise ValueError('ForceGello resampling requires unmodified joint or TCP delta actions')
        self.visual_latency_steps = int(visual_latency_steps)
        if self.visual_latency_steps != visual_latency_steps or self.visual_latency_steps < 0:
            raise ValueError('visual_latency_steps must be a non-negative integer')
        if sample_hz is None and (sample_phase is not None or self.visual_latency_steps):
            raise ValueError('sample_phase and visual latency require ForceGello sampling')

        action_key = [self.action_key]

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta.get('obs', dict())
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)

        self.zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        zarr_load_keys = set(action_key + rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys)
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        if sample_hz is None:
            replay_buffer = ReplayBuffer.copy_from_path(
                self.zarr_path, keys=zarr_load_keys)
        else:
            from ImplicitRDP.dataset.forcegello_sampling import load_joint_replay
            replay_buffer = load_joint_replay(
                self.zarr_path, zarr_load_keys, self.action_key, float(sample_hz),
                sample_phase=sample_phase)

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        self.relative_action = relative_action
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action
        self.rpy_for_rotation = rpy_for_rotation
        self.transforms = RealWorldTransforms(option=transform_params)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps * obs_temporal_downsample_ratio
        self.key_first_k = key_first_k

        self.val_ratio = val_ratio
        self.seed = seed
        self.max_train_episodes = max_train_episodes
        source_ids = getattr(replay_buffer, 'source_episode_ids', None)
        if source_ids is None:
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio, seed=seed)
            train_mask = downsample_mask(
                mask=~val_mask, max_n=max_train_episodes, seed=seed)
        else:
            # Keep all phases of one recorded episode in the same split.
            _, groups = np.unique(source_ids, return_inverse=True)
            group_val = get_val_mask(
                n_episodes=int(groups.max()) + 1, val_ratio=val_ratio, seed=seed)
            group_train = downsample_mask(
                mask=~group_val, max_n=max_train_episodes, seed=seed)
            val_mask = group_val[groups]
            train_mask = group_train[groups]

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k
            )
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
    
    def get_resume_params(self):
        return {
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
        }

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k=self.key_first_k
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys:
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            inter_gripper_data_dict.update({f"extended_{key}": list() for key in self.extended_lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key})
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    if 'extended_' in key:
                        inter_gripper_data_dict[key].append(data['extended_obs'][key.replace('extended_', '')])
                    else:
                        inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        if self.relative_action or self.rpy_for_rotation:
            # action and obs
            recalculate_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            # extended_obs
            recalculate_data_dict.update({f"extended_{key}": list() for key in self.extended_lowdim_keys if 'robot_tcp_pose' in key and 'wrt' not in key})
            for data in tqdm.tqdm(self, leave=False, desc='Calculating updated action/obs for normalizer'):
                for key in recalculate_data_dict.keys():
                    if key == 'action':
                        recalculate_data_dict[key].append(data[key])
                    elif 'extended_' in key:
                        recalculate_data_dict[key].append(data['extended_obs'][key.replace('extended_', '')])
                    else:
                        recalculate_data_dict[key].append(data['obs'][key])
            recalculate_data_dict = dict_apply(recalculate_data_dict, np.stack)

        # action
        if self.relative_action or self.rpy_for_rotation:
            action_all = recalculate_data_dict['action']
        else:
            action_all = self.replay_buffer[self.action_key][:, :self.shape_meta['action']['shape'][0]]
        normalizer['action'] = get_action_normalizer(action_all)

        # obs
        for key in self.lowdim_keys:
            if (self.relative_action or self.rpy_for_rotation) and key in recalculate_data_dict:
                normalizer[key] = get_action_normalizer(recalculate_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                if len(self.shape_meta['obs'][key]['shape']) == 1:
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
                elif len(self.shape_meta['obs'][key]['shape']) == 2:
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]])
                else:
                    raise ValueError(f"Only support 1d or 2d obs")
            else:
                if len(self.shape_meta['obs'][key]['shape']) == 1:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
                elif len(self.shape_meta['obs'][key]['shape']) == 2:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]])
                else:
                    raise ValueError(f"Only support 1d or 2d obs")

        for key in self.extended_lowdim_keys:
            if (self.relative_action or self.rpy_for_rotation) and f"extended_{key}" in recalculate_data_dict:
                normalizer[f"extended_{key}"] = get_action_normalizer(recalculate_data_dict[f"extended_{key}"])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[f"extended_{key}"] = get_action_normalizer(inter_gripper_data_dict[f"extended_{key}"])
            elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                if len(self.shape_meta['extended_obs'][key]['shape']) == 1:
                    normalizer[f"extended_{key}"] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                elif len(self.shape_meta['extended_obs'][key]['shape']) == 2:
                    normalizer[f"extended_{key}"] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0], :self.shape_meta['extended_obs'][key]['shape'][1]])
                else:
                    raise ValueError(f"Only support 1d or 2d extended obs")
            else:
                if len(self.shape_meta['extended_obs'][key]['shape']) == 1:
                    normalizer[f"extended_{key}"] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                elif len(self.shape_meta['extended_obs'][key]['shape']) == 2:
                    normalizer[f"extended_{key}"] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0], :self.shape_meta['extended_obs'][key]['shape'][1]])
                else:
                    raise ValueError(f"Only support 1d or 2d extended obs")

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        for key in self.extended_rgb_keys:
            normalizer[f"extended_{key}"] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def _observation_indices(self, key: str, sample_index: int, prefix_steps: int) -> np.ndarray:
        """Select the policy history, with camera frames held on the episode clock."""
        meta = self.shape_meta['obs'][key]
        stride = int(meta.get('temporal_downsample_ratio', self.obs_downsample_ratio))
        count = int(meta.get('n_obs_steps', prefix_steps))
        hold = int(meta.get('hold_steps', 1))
        if min(stride, count, hold) < 1:
            raise ValueError(f'invalid observation sampling for {key}')
        indices = prefix_steps - 1 - np.arange(count - 1, -1, -1) * stride
        if key in self.rgb_keys:
            indices = indices - self.visual_latency_steps
        if np.any(indices < 0) and key not in self.rgb_keys:
            raise ValueError(f'observation prefix is too short for {key}')
        if hold > 1:
            buffer_start, buffer_end, sample_start, _ = self.sampler.indices[sample_index]
            ends = self.replay_buffer.episode_ends[:]
            episode_index = np.searchsorted(ends, buffer_start, side='right')
            episode_start = 0 if episode_index == 0 else ends[episode_index - 1]
            absolute = np.clip(buffer_start + indices - sample_start, episode_start, buffer_end - 1)
            held = episode_start + ((absolute - episode_start) // hold) * hold
            indices = held - buffer_start + sample_start
        return indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        obs_downsample_ratio = self.obs_downsample_ratio

        obs_dict = dict()
        for key in self.rgb_keys:
            if self.forcegello_sampling:
                prefix_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon + self.n_latency_steps
                indices = self._observation_indices(key, idx, prefix_steps)
                if np.any(indices < 0):
                    buffer_start, _, sample_start, _ = self.sampler.indices[idx]
                    absolute = buffer_start + indices - sample_start
                    frames = np.stack([self.replay_buffer[key][int(i)] for i in absolute])
                else:
                    frames = data[key][indices]
            else:
                frames = data[key][:self.n_obs_steps][::-obs_downsample_ratio][::-1]
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(frames,-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                if self.forcegello_sampling:
                    prefix_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon + self.n_latency_steps
                    indices = self._observation_indices(key, idx, prefix_steps)
                    selected = data[key][indices]
                else:
                    selected = data[key][:self.n_obs_steps][::-obs_downsample_ratio][::-1]
                if len(self.shape_meta['obs'][key]['shape']) == 1:
                    obs_dict[key] = selected[:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
                elif len(self.shape_meta['obs'][key]['shape']) == 2:
                    obs_dict[key] = selected[:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]].astype(np.float32)
                else:
                    raise ValueError(f"Only support 1d or 2d obs")
                # save ram
                if key not in self.extended_lowdim_keys:
                    del data[key]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in ['left_robot_wrt_right_robot_tcp_pose', 'right_robot_wrt_left_robot_tcp_pose']:
            if key in obs_dict:
                if len(self.shape_meta['obs'][key]['shape']) == 1:
                    obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
                elif len(self.shape_meta['obs'][key]['shape']) == 2:
                    obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0], :self.shape_meta['obs'][key]['shape'][1]].astype(np.float32)
                else:
                    raise ValueError(f"Only support 1d or 2d obs for inter-gripper relative action")
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                if len(self.shape_meta['extended_obs'][key]['shape']) == 1:
                    extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                elif len(self.shape_meta['extended_obs'][key]['shape']) == 2:
                    extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0], :self.shape_meta['extended_obs'][key]['shape'][1]].astype(np.float32)
                else:
                    raise ValueError(f"Only support 1d or 2d extended obs")
                del data[key]

        action = data[self.action_key][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][self.n_obs_steps-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][self.n_obs_steps-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=obs_dict[key][self.n_obs_steps-1])

        if self.rpy_for_rotation:
            action = matrix_actions_to_rpy_actions(action)
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key:
                    obs_dict[key] = matrix_actions_to_rpy_actions(obs_dict[key])
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key:
                    extended_obs_dict[key] = matrix_actions_to_rpy_actions(extended_obs_dict[key])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy),
        }
        return torch_data
