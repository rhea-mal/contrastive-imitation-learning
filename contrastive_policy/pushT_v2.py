from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.1, beta_end=20.0):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._schedule_timesteps()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _schedule_timesteps(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def add_noise(self, trajectory, times):
        noise = torch.randn_like(trajectory)
        alphas_cumprod_t = self.alphas_cumprod[times].view(-1, 1, 1)
        noisy_trajectory = alphas_cumprod_t.sqrt() * trajectory + (1.0 - alphas_cumprod_t).sqrt() * noise
        return noisy_trajectory

    def step(self, model, trajectory, current_times, next_times, clamp=False):
        noise = torch.randn_like(trajectory)
        alpha_t = self.alphas[current_times].view(-1, 1, 1)
        alpha_next_t = self.alphas[next_times].view(-1, 1, 1)
        predicted_noise = model(trajectory, current_times)

        predicted_trajectory = (trajectory - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        new_trajectory = alpha_next_t.sqrt() * predicted_trajectory + (1.0 - alpha_next_t).sqrt() * noise

        if clamp:
            new_trajectory = torch.clamp(new_trajectory, -1.0, 1.0)

        return new_trajectory

    def sample_inital_position(self, condition_data, generator=None):
        return torch.randn_like(condition_data, generator=generator)

    def timesteps_to_times(self, timesteps):
        return timesteps.float() / self.timesteps

    def sample_times(self, trajectory, time_sampler='uniform'):
        B = trajectory.shape[0]
        if time_sampler == 'uniform':
            times = torch.randint(0, self.timesteps, (B,))
            return times, times
        else:
            raise ValueError(f"Unknown time sampler: {time_sampler}")

# Contrastive Decoding Policy

class ContrastiveDecodingPolicy(nn.Module):
    def __init__(self, shape_meta, noise_scheduler, horizon, n_action_steps, n_obs_steps, teacher_samples):
        super().__init__()
        self.shape_meta = shape_meta
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.teacher_samples = teacher_samples
        self.model = self._load_model()
        self.normalizer = None

    def _load_model(self):
        # Load your model here, for example:
        model = nn.Sequential(
            nn.Linear(self.shape_meta['input_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, self.shape_meta['output_dim'])
        )
        return model

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch['obs']
        action = batch['action']
        coverage = batch['coverage']
        alignment = batch['alignment']
        
        score = coverage + (1 - alignment / 180)

        score_min = score.min()
        score_max = score.max()
        score_norm = (score - score_min) / (score_max - score_min)

        pred_action = self.model(obs)
        teacher_action = self.teacher_samples['action'].detach()

        loss_pos = score_norm * F.mse_loss(pred_action, teacher_action, reduction='none').mean()
        loss_neg = (1 - score_norm) * F.mse_loss(pred_action, action, reduction='none').mean()

        loss = -torch.log(torch.exp(loss_pos) / (torch.exp(loss_pos) + torch.exp(loss_neg))).mean()
        return loss

class ConsensusSampling:
    def __init__(self, k: float):
        self.k = k

    def compute_consensus_score(self, demonstrations: List[Dict[str, torch.Tensor]]) -> List[float]:
        scores = [self._compute_demo_score(demo) for demo in demonstrations]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        normalized_scores = [(score - mean_score) / std_score for score in scores]
        return normalized_scores

    def _compute_demo_score(self, demo: Dict[str, torch.Tensor]) -> float:
        coverage = demo['coverage'].item()
        alignment = demo['alignment'].item()
        score = coverage + (1 - alignment / 180)
        return score

    def select_good_demos(self, demonstrations: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        consensus_scores = self.compute_consensus_score(demonstrations)
        mean_score = np.mean(consensus_scores)
        good_demos = [demo for demo, score in zip(demonstrations, consensus_scores) if score > mean_score + self.k * np.std(consensus_scores)]
        return good_demos

    def select_bad_demos(self, demonstrations: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        consensus_scores = self.compute_consensus_score(demonstrations)
        mean_score = np.mean(consensus_scores)
        bad_demos = [demo for demo, score in zip(demonstrations, consensus_scores) if score < mean_score - self.k * np.std(consensus_scores)]
        return bad_demos
    
class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def _calculate_coverage(self, sample) -> float:
        """
        Calculates the coverage of the T symbol in the PushT task.
        This function assumes the T symbol and the end-effector are represented in the sample.
        """
        # Example logic: Assuming sample['state'] contains the position of the T symbol
        # and sample['end_effector_pos'] contains the position of the end-effector.
        t_symbol_pos = sample['state'][..., :2]  # Assuming the first two elements are x, y of T symbol
        end_effector_pos = sample['end_effector_pos']  # Assuming this is provided in the sample
        
        # Calculate distance between T symbol and end-effector
        distance = np.linalg.norm(t_symbol_pos - end_effector_pos, axis=-1)
        
        # Coverage could be inversely proportional to the distance
        coverage = np.clip(1 - distance / np.max(distance), 0, 1)
        return coverage.mean()

    def _calculate_alignment(self, sample) -> float:
        """
        Calculates the alignment of the T symbol in the PushT task.
        This function assumes the T symbol and the end-effector orientations are represented in the sample.
        """
        # Example logic: Assuming sample['state'] contains the orientation of the T symbol
        # and sample['end_effector_orientation'] contains the orientation of the end-effector.
        t_symbol_orientation = sample['state'][..., 2]  # Assuming the third element is the orientation of T symbol
        end_effector_orientation = sample['end_effector_orientation']  # Assuming this is provided in the sample
        
        # Calculate the angular difference
        alignment = np.abs(t_symbol_orientation - end_effector_orientation)
        return alignment.mean()

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255
        end_effector_pos = sample['end_effector_pos']
        end_effector_orientation = sample['end_effector_orientation']

        coverage = self._calculate_coverage(sample)
        alignment = self._calculate_alignment(sample)

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
                'coverage': coverage,
                'alignment': alignment
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

dataset = PushTImageDataset(zarr_path='pusht.zarr', horizon=16)

normalizer = dataset.get_normalizer()

shape_meta = {'input_dim': 32, 'output_dim': 7}
noise_scheduler = NoiseScheduler()  # Initialize your noise scheduler
horizon = 1
n_action_steps = 1
n_obs_steps = 1

# Select good demonstrations as teacher samples
consensus_sampler = ConsensusSampling(k=2)
demonstrations = [dataset[i] for i in range(len(dataset))]
good_demos = consensus_sampler.select_good_demos(demonstrations)

# Use good demonstrations for the teacher samples
teacher_samples = {
    'obs': torch.stack([demo['obs'] for demo in good_demos]),
    'action': torch.stack([demo['action'] for demo in good_demos]),
    'coverage': torch.stack([demo['coverage'] for demo in good_demos]),
    'alignment': torch.stack([demo['alignment'] for demo in good_demos])
}

policy = ContrastiveDecodingPolicy(shape_meta, noise_scheduler, horizon, n_action_steps, n_obs_steps, teacher_samples)

policy.set_normalizer(normalizer)

for batch in DataLoader(dataset, batch_size=32, shuffle=True):
    loss = policy.compute_loss(batch) # Perform optimization step w computed loss
