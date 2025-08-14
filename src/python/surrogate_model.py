"""
surrogate_model.py

Neural network surrogate models (PyTorch) for apprximating finite-difference
solutions of nonlinear PDEs (initially Burgers' equation). Baseline provided
as a fully connected regressor; extensions may include convolutional or 
sequence models for spatiotemporal prediction.

Design goals:
- Fast inference relative to high-fidelity solvers.
- Flexible architecture specification (hidden sizes, activation, dropout).
- Clear I/O contract for downstream training and evaluation.
"""

from typing import List
import torch
import torch.nn as nn

class SurrogateNet(nn.Module):
    """
    Baseline fully connected surrogate for PDE solution mappings.

    Given a vectorized representation of inputs (e.g., initial condition values on a 
    grid and/or PDE parameters), the model predicts the corresponding solution
    field in a vectorized form.

    Parameters
    ----------
    input_dim : int
        Size of the input vector (e.g., nx for a single-time IC, or a reduced feature
        vector).
    output_dim : int
        Size of the output vector (e.g., nx for solution at a target time, or 
        (nt+1)*nx flattened).
    hidden_layers : list[int]
        Sizes of hidden layers in order (e.g., [512, 512, 256]).
    activation : torch.nn.Module, default nn.ReLU()
        Nonlinearity applied after each hidden layer.
    dropout : float, default 0.0
        Dropout probability for regularization; 0 disables dropout.

    Forward Input
    -------------
    x : torch.Tensor, shape (batch_size, output_dim)
        Input batch of vectorized features.
    
    Forward Output
    --------------
    torch.Tensor, shape (batch_size, output_dim)
        Predicted solution vector.
    
    Notes
    -----
    - This baseline model is agnostic to spatial structure; for grid-aware
      modeling, a CNN (1D conv) or U-Net-style architecture may be introduced
      later.
    - Loss functions and training loops are defined in 'training.py'.

    Examples
    --------
    >>> model = SurrogateNet(input_dim=256, output_dim=256, hidden_layers=[512, 512])
    >>> x = torch.randn(10, 256) 
    >>> y = model(x)  # calls .forward(x)
    >>> y.shape
    torch.Size([8, 256])
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: List[int],
            activation: nn.Module = nn.ReLU(),
            dropout: float = 0.0
    ):
        super().__init__()
        # Implementation intentionally deferred.
        raise NotImplementedError("Model construction pending (define layers in __init__).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input batch.
        
        Returns
        -------
        torch.Tensor, shape (batch_size, output_dim)
            Predicted batch outputs.        
        """
        # Implementation intentionally deferred.
        raise NotImplementedError("Forward pass pending (apply layers to x and return).")