# rl_agent/training/utils.py

import os
import torch
import random
import numpy as np
from typing import Optional, Any


def sample_hyperparameters(space: dict[str, list[float]]) -> dict[str, Any]:
    """
    Samples a new configuration from the defined hyperparameter space

    If the value is alist of two floats, sample log-uniformly between the two values.
    Otherwise, sample uniformly from the list of possible values.

    :param space: the hyperparameter space dictionary from the config file.
    :returns: a dictionary with concrete hyperparameter values.
    """
    config = {}
    for key, value in space.items():
        if len(value) == 2 and isinstance(value[0], float):
            config[key] = float(10**np.random.uniform(np.log10(value[0]), np.log10(value[1])))
        else:
            config[key] = random.choice(value)
    return config


def mutate_hyperparameters(config: dict[str, Any], space: dict[str, list[float]]) -> dict[str, Any]:
    """
    Perturbates one hyperparameter in an existing configuration

    A single hyperparameter is selected at random and perturbed by a factor of 0.8 or 1.2.
    For categorical hyperparameters, a new value is selected uniformly from the list of possible values

    :param config: the agent's current configuration dictionary.
    :param space: the search space dictionary from the config file.
    :returns: new mutated configuration dictionary.
    """
    mutated_config = config.copy()
    key_to_mutate = random.choice(list(space.keys()))
    value_space = space[key_to_mutate]
    if len(value_space) == 2 and isinstance(value_space[0], float):
        factor = random.choice([0.8, 1.2])
        new_value = mutated_config[key_to_mutate] * factor
        mutated_config[key_to_mutate] = max(value_space[0], min(new_value, value_space[1]))
    else:
        mutated_config[key_to_mutate] = random.choice(value_space)
    return mutated_config


def save_checkpoint(population: list[dict], generation: int, output_dir: str):
    """saves the entire state of the population to a file"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_gen_{generation}.pth")
    data_to_save = {'generation': generation, 'population': population}
    torch.save(data_to_save, checkpoint_path)
    print(f"checkpoint saved for generation {generation} at {checkpoint_path}")

def load_checkpoint(path: str) -> Optional[tuple[list[dict], int]]:
    """loads a population state from a checkpoint file"""
    if not os.path.exists(path):
        print(f"checkpoint file not found at {path}")
        return None
    checkpoint = torch.load(path)
    print(f"loaded checkpoint from generation {checkpoint['generation']} at {path}")
    return checkpoint['population'], checkpoint['generation']