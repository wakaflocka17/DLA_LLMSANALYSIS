import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    With this function we can set the PyTorch,
    Numpy and Python random seed.
    :param seed: int, random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)