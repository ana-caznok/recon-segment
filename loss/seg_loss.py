import torch
import torch.nn as nn

class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized multiplicative loss.

        Args:
            pred (torch.Tensor): Prediction tensor of shape (B, 1, Y, X)
            target (torch.Tensor): Target tensor of shape (B, 1, Y, X)

        Returns:
            torch.Tensor: Scalar tensor representing the loss
        """
        # Check tensor shapes and dimensionality
        if pred.shape != target.shape or pred.dim() != 4 or pred.shape[1] != 1:
            raise ValueError("Inputs must have shape (B, 1, Y, X) and match in shape.")

        # Unpack dimensions
        B, _, Y, X = pred.shape
        sum_pred = pred.sum(dim=(2, 3))
        sum_target = target.sum(dim=(2, 3))
        # Element-wise product and spatial sum
        # Efficiently compute sum over the last two dimensions (Y and X)
        # Keep the computation on the same device as the input tensors (CPU or GPU)
        loss = (target*pred).sum(dim=(2, 3)) / (sum_pred + sum_target)  # Shape: (B, 1)

        # Average over the batch dimension
        return 1 - loss.mean()
