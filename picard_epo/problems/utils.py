#!/usr/bin/env python3

import hashlib
import os
import pickle
import tempfile
import jax.numpy as jnp


def get_cache_path(data_hash, cache_type="optimal_solutions"):
    """Get cache file path in temp directory."""
    temp_dir = tempfile.gettempdir()
    cache_filename = f"{cache_type}_{data_hash}.pkl"
    return os.path.join(temp_dir, cache_filename)


def load_cache(data_hash, cache_type):
    """Load cached optimal solutions if they exist."""
    cache_path = get_cache_path(data_hash, cache_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
            return cached_data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None


def save_cache(data_hash, obj, cache_type):
    """Save optimal solutions to cache."""
    cache_path = get_cache_path(data_hash, cache_type)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved solutions to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving cache: {e}")


def create_problem_data_hash(config_data: dict):
    """Create a hash from problem generation config parameters for cache key generation."""
    sha = hashlib.sha256()
    sha.update(str(sorted(config_data.items())).encode())
    return str(sha.hexdigest())
