

# R2-DEAL-2 Agent Documentation

This document provides an overview of the R2-DEAL-2 agent, with its architecture, core components, and the process used for training and inference.

> Note from the authors:
> - Three students from this group are also enrolled in a RL course at TU/e.
> - This agent was developed as a way to test our knowledge on RL algorithms.

## Table of Contents
1.  [Overview](#1-overview)
2.  [File Structure](#2-file-structure)
3.  [Core Components (`agent_utils.py`)](#3-core-components-agent_utils)
4.  [The NFSP Agent Controller (`agent.py`)](#4-the-nfsp-agent-controller-agent)
5.  [The Training Pipeline (`training/`)](#5-the-training-pipeline-training)

---

## 1. Overview

`R2-DEAL-2` is an implementation of Neural Fictitious Self-Play (NFSP) [[1]](https://arxiv.org/abs/1603.01121), an algorithm designed to learn an approximate Nash Equilibrium in imperfect information games like Kuhn Poker.

The agent's learning process is built on two key components:
*   Its best-response policy is learned using a Double DQN (DDQN) [[2]](https://arxiv.org/abs/1509.06461) algorithm.
*   Exploration is managed by Noisy Networks [[3]](https://arxiv.org/abs/1706.10295), which add parametric noise to the neural network's weights, removing the need for a traditional epsilon-greedy strategy.

The final, agent was produced by a large-scale Population-Based Training (PBT) [[4]](https://arxiv.org/abs/1711.09846) run on an HPC cluster. This process simultaneously trained a population of agents and optimized their learning hyperparameters to find the most suitable strategy.

---

## 2. File Structure

The `rl_agent` module is organized to separate the core agent logic from the training and inference code.

```
/rl_agent/
│
├── agent.py              # the main NFSPAgent class for training
├── agent_utils.py        # the building blocks: networks, layers, and buffers
├── inference.py          # the final, R2-DEAL-2 agent for competition
├── policy_3_card.pth     # the trained weights for the 3-card agent
├── policy_4_card.pth     # the trained weights for the 4-card agent
├── README.md             # you are here!
│
└───/training/
    ├── train.py          # the main PBT orchestrator script for training
    ├── config.yaml       # config file for the training script
    ├── evaluation.py     # module for evaluating a trained population
    ├── kuhn_env.py       # a parallelizable poker simulator for training
    ├── benchmark_agents.py # CFR and Random agents to train against
    └── utils.py          # helper functions for the PBT process
```

---

## 3. Core Components (`agent_utils.py`)

This module defines the building blocks of the agent.

*   **`NoisyLinear` Layer**
    This is the agent exploration mechanism, which replaces the need for an epsilon-greedy strategy. It adds learnable noise directly to the network's weights, pushing the agent to try new actions during training.

*   **`BestResponseNet`**
    This network is the agent's exploratory logic. It uses the `NoisyLinear` layers and is actively trained with Reinforcement Learning (RL) to discover the best counter-moves to an opponent strategy.

*   **`AverageStrategyNet`**
    This is the agent's stable memory. It is a deterministic network trained to imitate the historical average of the `BestResponseNet`'s actions. This network provides the final, robust policy that is used for tournament play.

*   **`ReplayBuffer`**
    A standard memory buffer that stores recent game transitions (`state`, `action`, `reward`, etc.) to train the `BestResponseNet`.

*   **`ReservoirBuffer`**
    A more custom buffer used to train the `AverageStrategyNet`. It keeps a uniform random sample of actions from the entire training history, making sure the average strategy is unbiased and stable.

---

## 4. NFSP Agent Controller (`agent.py`)

This module combines the core components from `agent_utils.py` into a single `NFSPAgent` class. This class is the controller responsible for the agent's learning process during training.

The main point to the NFSP algorithm lies in how the agent decides its actions, which is controlled by the `act` method and a hyperparameter called `eta`:

*   With a probability of `eta`, the agent acts according to its **`AverageStrategyNet`**. This represents the agent exploiting its current best guess of the optimal average strategy.
*   Otherwise, it acts according to its exploratory **`BestResponseNet`**. This represents the agent exploring and learning a better counter-strategy.

By a mix of these two, the agent learns a stable average strategy while exploring for better counter-strategies, allowing it to converge towards near-optimal policy (in theory...).

---

## 5. The Training Pipeline (`training/`)

The `training/` subdirectory contains the pipeline used to train the R2-DEAL-2 from scratch.

### The Environment (`kuhn_env.py`)

This is a fast, and "match-aware" simulator for Kuhn Poker. It is intentionally decoupled from the tournament's official `GameState` to enable faster simulation speeds.
This design is critical for the massive parallelization required by our training process. The environment tracks player bankrolls, and the agent's reward is the direct change in its bankroll after a hand, encouraging it to learn a winning match strategy.

### The Orchestrator (`train.py`)

This is the main PBT script that manages a population of `NFSPAgent`s. It orchestrates the entire training lifecycle:
1.  Parallel Training: It spawns a pool of worker processes, one for each agent in the population. Each worker trains its agent by playing thousands of matches against other agents and benchmark opponents.
2.  Evaluation: At the end of a training generation, it evaluates each agent's performance based on its match win rate.
3.  Evolution: It replaces the worst-performing agents with mutated copies of the top performers, the population evolves towards better hyperparameters and strategies over time.

### Configuration (`config.yaml`)

The entire training process is controlled by a single YAML configuration file. The final configuration used for training the 4-card agent is shown below.

```yaml
# environment parameters
env_params:
  num_cards: 4                          # 4-card kuhn poker.
  starting_bankroll: 5                  # initial bank of each player

# PBT parameters
pbt_params:
  population_size: 40                   # number of parallel agents (N-1 or N-2 CPU cores)
  generations: 250                      # number of pbt training-evaluation-evolution loops
  matches_per_generation: 2500          # number of full matches each agent plays per generation
  benchmark_match_prob: 0.3             # prob. of an agent playing against a fixed benchmark (cfr/random) instead of another nfsp agent
  exploit_top_k_percent: 20             # the top 20% of agents are candidates to be copied
  exploit_bottom_k_percent: 20          # the bottom 20% of agents are replaced each generation.

# periodic evaluation parameters 
evaluation_params:
  evaluate_every: 10                    # run a full evaluation of the champion every 10 generations.
  internal_matches_per_pair: 200        # number of hands to determine the champion in the internal tournament
  benchmark_matches: 500                # number of hands to play against each benchmark (cfr/random) in the evaluation

# agent hyperparameter search space 
hyperparameter_space:
  hidden_size: [64, 128, 256]           # choices for the number of neurons in hidden layers
  batch_size: [128]                     # fixed batch size for network updates
  rl_learning_rate: [0.0001, 0.005]     # log-uniform range for the dqn learning rate
  sl_learning_rate: [0.0001, 0.005]     # log-uniform range for the supervised learning rate
  gamma: [0.95, 0.999]                  # discount factor for future rewards.
  tau: [0.001, 0.01]                    # soft update rate for the target network
  gradient_clip: [1.0, 10.0]            # range for max gradient norm to prevent exploding gradients
  eta: [0.05, 0.2]                      # nfsp: probability of using the average strategy policy during training
  rl_buffer_size: [200000, 500000]      # range for the size of the rl experience replay buffer
  sl_buffer_size: [500000, 1000000]     # range for the size of the supervised learning reservoir buffer

# checkpointing
checkpointing:
  output_dir: "agentx/training/training_outs/4_card_run" # directory to save training progress.
  save_every_n_generations: 5           # frequency of saving checkpoints.
```

### HPC Execution & SLURM Script

Training is computationally intensive and is designed to run on a multi-core HPC node. The script should be launched as a module from the project root to ensure all imports resolve correctly:
`python -m rl_agent.training.train --config rl_agent/training/config.yaml`

The following is an example SLURM batch script used to submit the training job.

```bash
#!/bin/bash

#SBATCH --job-name=poker-pbt          
#SBATCH --partition=tue.default.q     # the partition to run on, we prefer tue.default.q as one node has ~98 cores available.
#SBATCH --nodes=1                     # request one node
#SBATCH --ntasks=1                    # request one task per node
#SBATCH --cpus-per-task=42            # request 40 cpus, matching population_size + 2 for the manager
#SBATCH --mem=32G                     # requesting memory/ram
#SBATCH --time=48:00:00               # 48 hours of wall time
#SBATCH --output=slurm-%j.out         # log file 


#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.bekishoglu@student.tue.nl

# job execution:
module purge
module load Python/3.10.4-GCCcore-11.3.0  # load the required python module on the hpc
source .venv/bin/activate                 # activate the project's virtual environment

# run the main pbt training script
stdbuf -oL python -m rl_agent.training.train --config rl_agent/training/config.yaml
```


