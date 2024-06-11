from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from torch.utils.data import DataLoader, Subset

class ContrastiveDecodingPolicy(BaseImagePolicy):
    def __init__(self, shape_meta, noise_scheduler, horizon, n_action_steps, n_obs_steps, teacher_path):
        super().__init__()
        self.shape_meta = shape_meta
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.teacher_path = teacher_path
        self.model = self._load_model()
        self.teacher = self._load_teacher_model(teacher_path)
        self.normalizer = None

    def _load_model(self):
        # Load your model here, for example:
        model = nn.Sequential(
            nn.Linear(self.shape_meta['input_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, self.shape_meta['output_dim'])
        )
        return model

    def _load_teacher_model(self, teacher_path):
        # Load the teacher model here
        teacher_model = torch.load(teacher_path)
        return teacher_model

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        obs = batch['obs']
        action = batch['action']
        score = batch['coverage'] + (1 - batch['alignment'] / 180)

        score_min = score.min()
        score_max = score.max()
        score_norm = (score - score_min) / (score_max - score_min)

        pred_action = self.model(obs)
        teacher_action = self.teacher(obs).detach()

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

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

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
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
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
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


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
            raise ValueError(f"Unknown sampler: {time_sampler}")

    
# Initialize dataset
dataset = PushTLowdimDataset(zarr_path='pusht.zarr', horizon=1, pad_before=0, pad_after=0, obs_key='keypoint', state_key='state', action_key='action')

# Get normalizer
normalizer = dataset.get_normalizer()

# Initialize policy
shape_meta = {'input_dim': 32, 'output_dim': 7}  # Example meta data
noise_scheduler = NoiseScheduler()  # Initialize your noise scheduler
horizon = 1
n_action_steps = 1
n_obs_steps = 1
teacher_path = 'path_to_teacher_model.pth'
policy = ContrastiveDecodingPolicy(shape_meta, noise_scheduler, horizon, n_action_steps, n_obs_steps, teacher_path)

# Set normalizer
policy.set_normalizer(normalizer)

# Train policy with dataset
for batch in DataLoader(dataset, batch_size=32, shuffle=True):
    loss = policy.compute_loss(batch)
    # Perform optimization step with the computed loss

# Consensus sampling
consensus_sampler = ConsensusSampling(k=2)  # Example k value
demonstrations = [dataset[i] for i in range(len(dataset))]
good_demos = consensus_sampler.select_good_demos(demonstrations)
bad_demos = consensus_sampler.select_bad_demos(demonstrations)