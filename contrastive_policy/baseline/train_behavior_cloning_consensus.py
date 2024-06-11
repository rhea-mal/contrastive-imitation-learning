from typing import Dict, List
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

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
        t_symbol_pos = sample['state'][..., :2]
        end_effector_pos = sample['end_effector_pos']
        distance = np.linalg.norm(t_symbol_pos - end_effector_pos, axis=-1)
        coverage = np.clip(1 - distance / np.max(distance), 0, 1)
        return coverage.mean()

    def _calculate_alignment(self, sample) -> float:
        t_symbol_orientation = sample['state'][..., 2]
        end_effector_orientation = sample['end_effector_orientation']
        alignment = np.abs(t_symbol_orientation - end_effector_orientation)
        return alignment.mean()

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32)
        image = np.moveaxis(sample['img'],-1,1)/255
        end_effector_pos = sample['end_effector_pos']
        end_effector_orientation = sample['end_effector_orientation']

        coverage = self._calculate_coverage(sample)
        alignment = self._calculate_alignment(sample)

        data = {
            'obs': {
                'image': image,
                'agent_pos': agent_pos,
                'coverage': coverage,
                'alignment': alignment
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def compute_consensus_score(self, demonstrations: List[Dict[str, torch.Tensor]]) -> List[float]:
        scores = []
        for demo in demonstrations:
            coverage = demo['obs']['coverage']
            alignment = demo['obs']['alignment']
            score = coverage + (1 - alignment / 180)
            scores.append(score)
        return scores

    def select_good_demos(self, demonstrations: List[Dict[str, torch.Tensor]], k: float) -> List[Dict[str, torch.Tensor]]:
        scores = self.compute_consensus_score(demonstrations)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + k * std_score
        good_demos = [demo for demo, score in zip(demonstrations, scores) if score > threshold]
        return good_demos


class SimpleNNPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNNPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_policy(policy, dataloader, optimizer, criterion, epochs=10):
    policy.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            obs = batch['obs']['image'].view(batch['obs']['image'].size(0), -1)
            actions = batch['action']
            predictions = policy(obs)
            loss = criterion(predictions, actions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


if __name__ == "__main__":
    zarr_path = 'pusht.zarr'
    dataset = PushTImageDataset(zarr_path, horizon=1, pad_before=0, pad_after=0)
    normalizer = dataset.get_normalizer()
    demonstrations = [dataset[i] for i in range(len(dataset))]
    good_demos = dataset.select_good_demos(demonstrations, k=1.0)

    good_demos_dataset = torch.utils.data.Dataset()
    good_demos_dataset.data = good_demos

    dataloader = DataLoader(good_demos_dataset, batch_size=32, shuffle=True)

    input_dim = good_demos[0]['obs']['image'].numel()
    output_dim = good_demos[0]['action'].shape[-1]
    policy = SimpleNNPolicy(input_dim, output_dim)

    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_policy(policy, dataloader, optimizer, criterion, epochs=10)