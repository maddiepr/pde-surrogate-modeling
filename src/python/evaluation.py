"""
evaluation.py

Evaluation utilities for surrogate models versus finite-difference (FD) 
reference solutions. Provides a single entry point 'evaluate_surrogate' for
computing quantitative error metrics and a clear report suitable for
experiment tracking.

Design goals:
- Consistent, lightweight interface for evaluating trained models.
- Extensible metrics registry (caller can pass additional metric callables).
- Device-agnostic inference (CPU/GPU) with no training side effects.
"""

from typing import Dict, Callable, Optional
import numpy as np
import torch

def evaluate_surrogate(
        model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
        device: str = "cpu",
        reduce: str = "mean",
) -> Dict[str, float]:
    """
    Evaluate a trained surrogate model on held-out data.

    Parameters
    ----------
    model: torch.nn.Module
        Trained model. This function sets the model to eval mode during inference.
    inputs: torch.Tensor
        Input tensor of shape (N, input_dim, ...) matching the model's expected input.
    targets: torch.Tensor
        Ground-truth tensor aligned with 'inputs', shape (N, output_dim, ...)    
    metrics: dict[str, callable] or None, default None
        Optional mapping of metric names to functions with signature
        '(pred: torch.Tensor, target: torch.Tensor) -> float'.
        Useful for L1, relative L2, max error, etc.
    device: {"cpu", "cuda"} or str, default "cpu"
        Device for inference.    
    reduce: {"mean", "none"}, default "mean"
        Reduction for the built-in MSE reported as "mse".
        - "mean": average over the batch and all output dims
        - "none": no reduction; returns NaN for "mse" and relies on custom metrics

    Returns
    -------
    report: dict[str, float]
        Dictionary of computed metrics. Always includes:
            - "mse": float  # mean squared error if reduce="mean", NaN otherwise
        Plus any additonal '"metric/<name>"' entries from 'metrics'.

    Notes
    -----
    - Gradients are disabled during evaluation.
    - Inputs and targets are not shuffled or batched here; pass pre-batched tensors
      if desired. For large datasets, consider iterating in a DataLoader externally.
    - This function does not modify model parameters.

    Examples
    --------
    >>> model.eval()
    >>> with torch.no_grad():
    ...     report = evaluate_surrogate(model, inputs, targets, device="cpu",
    ...                                 metrics={"l1": lambda p,t: (p-t).abs().mean().item()})
    >>> "mse" in report
    True
    """
    raise NotImplementedError("Evaluation implementation pending.")