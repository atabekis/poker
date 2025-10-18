# src/utils.py

import os
import torch
import random
import numpy as np
from typing import Any, Optional


def sample_hyperparameters(space: dict[str, list[float]]) -> dict[str, Any]:
    """
    samples a new configuration from the defined hyperparameter space.

    :param space: the hyperparameter space dictionary from the config file.
    :returns: a dictionary with concrete hyperparameter values.
    """
    config = {}
    for key, value in space.items():
        if len(value) == 2 and isinstance(value[0], float):
            # continuous range, sample log-uniformly
            min_val, max_val = value
            config[key] = float(10 ** np.random.uniform(np.log10(min_val), np.log10(max_val)))
        else:
            # discrete choices
            config[key] = random.choice(value)
    return config


def mutate_hyperparameters(config: dict[str, Any], space: dict[str, list[float]]) -> dict[str, Any]:
    """
    perturbs one hyperparameter in an existing configuration.

    :param config: the agent's current configuration dictionary.
    :param space: the hyperparameter space dictionary from the config file.
    :returns: a new, mutated configuration dictionary.
    """
    mutated_config = config.copy()
    key_to_mutate = random.choice(list(space.keys()))

    value_space = space[key_to_mutate]

    if len(value_space) == 2 and isinstance(value_space[0], float):
        # perturb continuous value by a factor of 0.8 or 1.2
        factor = random.choice([0.8, 1.2])
        new_value = mutated_config[key_to_mutate] * factor
        # clamp the new value to be within the defined space
        min_val, max_val = value_space
        mutated_config[key_to_mutate] = max(min_val, min(new_value, max_val))
    else:
        # re-sample from discrete choices
        mutated_config[key_to_mutate] = random.choice(value_space)

    return mutated_config


def save_checkpoint(population: list[dict], generation: int, output_dir: str):
    """
    saves the entire state of the population to a file.

    :param population: a list of dictionaries, where each dict represents an agent's state.
    :param generation: the current generation number.
    :param output_dir: the directory to save the checkpoint in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_path = os.path.join(output_dir, f"checkpoint_gen_{generation}.pth")

    # an agent's state includes its config, weights, and performance
    data_to_save = {
        'generation': generation,
        'population': population
    }

    torch.save(data_to_save, checkpoint_path)
    print(f"\ncheckpoint saved for generation {generation} at {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[tuple[list[dict], int]]:
    """
    loads a population state from a checkpoint file.

    :param checkpoint_path: the path to the .pth checkpoint file.
    :returns: a tuple of (population, generation) if the file exists, otherwise none.
    """
    if not os.path.exists(checkpoint_path):
        print(f"checkpoint file not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path)
    population = checkpoint['population']
    generation = checkpoint['generation']

    print(f"loaded checkpoint from generation {generation} at {checkpoint_path}")
    return population, generation