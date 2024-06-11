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

## Lerobot Usage

### Visualize Dataset

To visualize a specific episode from the Push-T dataset, use the following command:

```bash
python lerobot/scripts/visualize_dataset.py --repo-id lerobot/pusht --episode-index 0
