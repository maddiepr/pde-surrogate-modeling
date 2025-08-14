"""
training.py

Supervised training utilities for neural surrogate models. Provides a 
single entry point 'train_surrogate' that performs batched training with
optional validation, checkpointing, and basic metric logging.

Design goals:
- Minimal explicit API compatible with PyTorch workflows.
- Reproducible runs (caller handles seeding in utils.set_seed).
- Clear separation of concerns: data preparation is external; this 
  module only consumes DataLoaders and a configured model/optimizer/loss.
"""

from typing import Callable, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader

def train_surrogate(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epochs: int,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
        grad_clip: Optional[float] = None,
        log_interval: int = 100,
) -> Dict[str, Any]:
    """
    Train a surrogate model on PDE datasets with optional validation and
    checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to train. Should define .train() / .eval() behavior and
        valid forward pass.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding (inputs, targets) for training.
    val_loader : torch.utils.data.DataLoader or None
        Optional DataLoader yielding (inputs, targets) for validation each epoch.
        If None, no validation metrics are computed.
    optimizer : torch.optim.Optimizer
        Configured optimizer instance (e.g., Adam).
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function mapping (predictions, targets) -> scalar loss tensor
        (e.g., MSELoss()).
    epochs : int
        Number of full passes over the training dataset.
    device : {"cpu", "cuda"} or str, default "cpu"
        Device on which training/inference is executed.
    checkpoint_path : str or None, default None
        If provided, the best model (by lowest validation loss) is saved to this path
        via 'torch.save(model.state_dic(), checkpoint_path)'.
    grad_clip : float or None, default None
        If provided, clip gradient norm to this value (e.g., 1.0) before optimizer step.
    log_interval : int, default 100
        Print/record training loss every 'log_interval' iterations (caller may redirect logs).
    
    Returns
    -------
    history : dict
        Dictionary containing basic training curves and summary:
            - "training_loss": list[float]      # average loss per epoch
            - "val_loss": list[float] or []     # average validation loss per epoch (if val_loader is not None)
            - "best_val_loss": float or None    # best validation loss observed
            - "epochs": int                     # number of epochs completed
    
    Notes
    -----
    - This function does not alter random seeds; ensure reproducibility externally.
    - Mixed precision, learning rate scheduling, early stopping, and advanced
      logging can be layered on top by the caller or added later without changing this API.
    - Expected input/output shapes should align with the model and dataset:
        inputs:  (batch_size, input_dim, ...)
        targets: (batch_size, output_dim, ...)
      Exact dimensions depend on how the dataset encodes PDE solutions.

    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> model = ...  # nn.Module
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> loss_fn = torch.nn.MSELoss()
    >>> train_ds = TensorDataset(torch.randn(256, 128), torch.randn(256, 128))
    >>> val_ds = TensorDataset(torch.randn(64, 128), torch.randn(64, 128))
    >>> train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    >>> val_loader = DataLoader(val_ds, batch_size=32)
    >>> history = train_surrogate(
    ...     model, train_loader, val_loader, optimizer, loss_fn, epochs=10, device="cpu"
    ... )
    """
    raise NotImplementedError("Training loop implementation pending.")