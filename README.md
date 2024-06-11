# Consensus Sampling in Contrastive Decoding for Imitation Learning in Robotics


https://github.com/rhea-mal/contrastive-imitation-learning/assets/70975260/e042b7f5-fd17-4d57-bd38-dbea6049bef0


## Overview

We leverage inter-sample relationships in demonstration quality using consensus sampling for contrastive imitation learning for robotics applications. Our contrastive + consensus sampling method, based on Consistency Policy, outperforms baseline behavior cloning approaches in both vanilla and consensus selected behavior cloning. We successfully integrated consensus sampling with contrastive learning for behavior cloning, demonstrating the viability of our approach for annotating human demonstrations. Additionally, we incorporated noise to handle periodic variance in rewards, effectively maintaining performance despite temporal correlations. Experiments showed that binary and continuous scoring methods yield similar performance in the PushT task, with a final 92.4\% success rate. Future research directions include pairwise mapping of similar states to good and bad demonstrations, expanding the approach to more tasks, and implementing online reinforcement learning.


## Features

- **Dataset Visualization**: Tools to visualize episodes from the Push-T dataset.
- **Model Training**: Frameworks for training models using behavior cloning and contrastive imitation learning.
- **Consensus Sampling**: Implementation of consensus sampling to filter good and bad demonstrations based on specified metrics.
- **Metrics Calculation**: Functions to calculate coverage and alignment for the Push-T task.

## Installation

### Prerequisites

- Python 3.8 or later
- Lerobot (https://github.com/huggingface/lerobot)
- Diffusion Policy Push T demonstration dataset (https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/)
- pip (Python package installer)

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/rhea-mal/contrastive-imitation-learning.git
    cd contrastive-imitation-learning
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

# Contrastive Decoding Policy for PushT v2 Dataset

This project implements a contrastive decoding policy for imitation learning using the PushT v2 dataset. The policy leverages good demonstrations to guide the learning process and uses a noise scheduler to simulate variability in the data.

## Key Components

### NoiseScheduler
The `NoiseScheduler` class manages the addition of noise to the trajectory data and the scheduling of diffusion steps.

#### Methods:
- `__init__(self, timesteps=1000, beta_start=0.1, beta_end=20.0)`: Initializes the noise scheduler with specified timesteps and beta values.
- `_schedule_timesteps(self)`: Creates a linear schedule for the beta values.
- `add_noise(self, trajectory, times)`: Adds noise to the trajectory based on specified times.
- `step(self, model, trajectory, current_times, next_times, clamp=False)`: Advances the trajectory by one step in the diffusion process.
- `sample_inital_position(self, condition_data, generator=None)`: Samples an initial random position for the trajectory.
- `timesteps_to_times(self, timesteps)`: Converts timesteps to a normalized time format.
- `sample_times(self, trajectory, time_sampler='uniform')`: Samples times for the diffusion process.

### ContrastiveDecodingPolicy
The `ContrastiveDecodingPolicy` class implements the contrastive decoding policy for imitation learning.

#### Methods:
- `__init__(self, shape_meta, noise_scheduler, horizon, n_action_steps, n_obs_steps, teacher_samples)`: Initializes the policy with metadata, a noise scheduler, horizon, action steps, observation steps, and teacher samples.
- `_load_model(self)`: Loads the model architecture.
- `set_normalizer(self, normalizer)`: Sets the normalizer for the policy.
- `compute_loss(self, batch)`: Computes the loss for a batch of data using contrastive terms weighted by normalized scores.

### ConsensusSampling
The `ConsensusSampling` class identifies good and bad demonstrations based on a consensus score.

#### Methods:
- `__init__(self, k)`: Initializes the sampler with a standard deviation factor `k`.
- `compute_consensus_score(self, demonstrations)`: Computes the consensus scores for the demonstrations.
- `select_good_demos(self, demonstrations)`: Selects good demonstrations based on consensus scores.
- `select_bad_demos(self, demonstrations)`: Selects bad demonstrations based on consensus scores.

### PushTImageDataset
The `PushTImageDataset` class loads and processes the PushT v2 dataset.

#### Methods:
- `__init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0, seed=42, val_ratio=0.0, max_train_episodes=None)`: Initializes the dataset with specified parameters.
- `get_validation_dataset(self)`: Returns the validation dataset.
- `get_normalizer(self, mode='limits', **kwargs)`: Returns a normalizer for the dataset.
- `_calculate_coverage(self, sample)`: Calculates the coverage of the T symbol in the PushT task.
- `_calculate_alignment(self, sample)`: Calculates the alignment of the T symbol in the PushT task.
- `__len__(self)`: Returns the length of the dataset.
- `_sample_to_data(self, sample)`: Converts a sample to a data dictionary.
- `__getitem__(self, idx)`: Gets a data sample by index.

# Initialize dataset
dataset = PushTImageDataset(zarr_path='pusht.zarr', horizon=16)
normalizer = dataset.get_normalizer()

shape_meta = {'input_dim': 32, 'output_dim': 7}
noise_scheduler = NoiseScheduler()

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

# Initialize policy
policy = ContrastiveDecodingPolicy(shape_meta, noise_scheduler, horizon=1, n_action_steps=1, n_obs_steps=1, teacher_samples=teacher_samples)
policy.set_normalizer(normalizer)
for batch in DataLoader(dataset, batch_size=32, shuffle=True):
    loss = policy.compute_loss(batch)
    # Perform optimization step with the computed loss


## Lerobot Usage - Visualize Dataset

To visualize a specific episode from the Push-T dataset, use the following command:

```bash
python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0
