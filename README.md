# Consensus Sampling in Contrastive Decoding for Imitation Learning in Robotics

## Overview

LeRobot is a robotics simulation and visualization tool designed to work with the Push-T dataset. This project aims to provide a framework for visualizing, analyzing, and training robotic behavior models using the Push-T dataset. The project includes tools for dataset management, model training, and visual evaluation of robotic tasks.

## Features

- **Dataset Visualization**: Tools to visualize episodes from the Push-T dataset.
- **Model Training**: Frameworks for training models using behavior cloning and contrastive imitation learning.
- **Consensus Sampling**: Implementation of consensus sampling to filter good and bad demonstrations based on specified metrics.
- **Metrics Calculation**: Functions to calculate coverage and alignment for the Push-T task.

## Installation

### Prerequisites

- Python 3.8 or later
- Lerobot
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
