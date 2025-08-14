"""
test_utils.py

Unit tests for utility functions in utils.py
"""

from src.python.utils import set_seed
import numpy as np
import torch

import sys, os
print("PYTHONPATH head:", sys.path[0])
print("Expect src at:", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_set_seed_numpy_and_torch_repro():
    """ 
    Ensure set_seed() produced reproducible random numbers.
    """
    set_seed(123)
    a1 = np.random.rand(3)
    t1 = torch.rand(3)

    set_seed(123)
    a2 = np.random.rand(3)
    t2 = torch.rand(3)

    # Should be exactly equal
    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)