import random
import numpy as np

def set_global_determinism(seed: int = 1337):
    """Set global random seeds for deterministic behavior"""
    random.seed(seed)
    np.random.seed(seed)