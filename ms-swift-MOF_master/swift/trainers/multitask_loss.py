# Copyright (c) Alibaba, Inc. and its affiliates.
# Multi-task learning loss weighting strategies
#
# References:
# - Uncertainty Weighting: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
# - Dynamic Weight Average: Liu et al., "End-to-End Multi-Task Learning with Attention" (CVPR 2019)
# - GradNorm: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (ICML 2018)

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal
from collections import deque

from swift.utils import get_logger

logger = get_logger()


class MultiTaskLossWeighter(nn.Module):
    """
    Multi-task loss weighting module supporting multiple strategies.
    
    Strategies:
    - 'fixed': Fixed weights (default behavior)
    - 'uncertainty': Learned uncertainty-based weighting (Kendall et al., 2018)
    - 'dwa': Dynamic Weight Average based on loss change rate (Liu et al., 2019)
    
    Usage:
        weighter = MultiTaskLossWeighter(strategy='uncertainty', task_names=['cls', 'reg'])
        total_loss = weighter(cls_loss=cls_loss, reg_loss=reg_loss)
    """
    
    def __init__(
        self,
        strategy: Literal['fixed', 'uncertainty', 'dwa'] = 'uncertainty',
        task_names: list = None,
        initial_weights: Dict[str, float] = None,
        dwa_temperature: float = 2.0,
        dwa_window_size: int = 2,
        device: str = 'cuda',
    ):
        super().__init__()
        self.strategy = strategy
        self.task_names = task_names or ['cls', 'reg']
        self.dwa_temperature = dwa_temperature
        self.device = device
        
        # Initialize based on strategy
        if strategy == 'uncertainty':
            # Learnable log variance parameters (sigma^2)
            # Loss = (1/2σ²) * L + log(σ) for regression
            # Loss = (1/σ²) * L + log(σ) for classification
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1, device=device))
                for name in self.task_names
            })
            logger.info(f"Initialized Uncertainty Weighting for tasks: {self.task_names}")
            
        elif strategy == 'dwa':
            # Dynamic Weight Average: track loss history
            self.loss_history = {name: deque(maxlen=dwa_window_size) for name in self.task_names}
            self.current_weights = {name: 1.0 for name in self.task_names}
            logger.info(f"Initialized DWA Weighting (T={dwa_temperature}) for tasks: {self.task_names}")
            
        elif strategy == 'fixed':
            self.fixed_weights = initial_weights or {name: 1.0 for name in self.task_names}
            logger.info(f"Using fixed weights: {self.fixed_weights}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'fixed', 'uncertainty', 'dwa'")
    
    def forward(
        self,
        cls_loss: Optional[torch.Tensor] = None,
        reg_loss: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            cls_loss: Classification loss (CrossEntropy)
            reg_loss: Regression loss (MSE)
            
        Returns:
            Weighted total loss
        """
        losses = {'cls': cls_loss, 'reg': reg_loss}
        losses.update(kwargs)  # Support additional tasks
        
        # Filter out None losses
        active_losses = {k: v for k, v in losses.items() if v is not None}
        
        if not active_losses:
            return torch.tensor(0.0, requires_grad=True)
        
        if self.strategy == 'uncertainty':
            return self._uncertainty_weighting(active_losses)
        elif self.strategy == 'dwa':
            return self._dwa_weighting(active_losses)
        else:  # fixed
            return self._fixed_weighting(active_losses)
    
    def _uncertainty_weighting(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Uncertainty-based weighting (Kendall et al., 2018).
        
        For classification: L_cls / (2 * σ_cls²) + log(σ_cls)
        For regression: L_reg / (2 * σ_reg²) + log(σ_reg)
        
        The model learns σ (uncertainty) for each task.
        Higher uncertainty → lower weight.
        """
        total_loss = 0.0
        
        for name, loss in losses.items():
            if name not in self.log_vars:
                # Add new task dynamically
                self.log_vars[name] = nn.Parameter(
                    torch.zeros(1, device=loss.device)
                )
            
            log_var = self.log_vars[name]
            
            # Precision (inverse variance): exp(-log_var) = 1/σ²
            precision = torch.exp(-log_var)
            
            if name == 'reg':
                # Regression: 0.5 * precision * loss + 0.5 * log_var
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            else:
                # Classification: precision * loss + log_var
                weighted_loss = precision * loss + log_var
            
            total_loss = total_loss + weighted_loss.squeeze()
        
        return total_loss
    
    def _dwa_weighting(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Dynamic Weight Average (Liu et al., 2019).
        
        Weight based on relative loss decrease rate:
        w_k(t) = K * exp(r_k(t-1) / T) / Σ exp(r_j(t-1) / T)
        
        where r_k = L_k(t-1) / L_k(t-2) is the loss ratio.
        """
        K = len(losses)
        
        # Update loss history and compute weights
        for name, loss in losses.items():
            self.loss_history[name].append(loss.detach().item())
        
        # Need at least 2 steps to compute ratios
        if all(len(h) >= 2 for h in self.loss_history.values()):
            # Compute loss ratios
            ratios = {}
            for name in losses.keys():
                history = list(self.loss_history[name])
                # r_k = L(t-1) / L(t-2)
                ratios[name] = history[-1] / (history[-2] + 1e-8)
            
            # Softmax with temperature
            exp_ratios = {name: torch.exp(torch.tensor(r / self.dwa_temperature)) 
                         for name, r in ratios.items()}
            sum_exp = sum(exp_ratios.values())
            
            self.current_weights = {
                name: (K * exp_r / sum_exp).item() 
                for name, exp_r in exp_ratios.items()
            }
        
        # Compute weighted loss
        total_loss = sum(
            self.current_weights.get(name, 1.0) * loss 
            for name, loss in losses.items()
        )
        
        return total_loss
    
    def _fixed_weighting(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fixed weight combination."""
        total_loss = sum(
            self.fixed_weights.get(name, 1.0) * loss 
            for name, loss in losses.items()
        )
        return total_loss
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights for logging."""
        if self.strategy == 'uncertainty':
            # Convert log_var to actual weight: 1 / (2 * σ²) = 0.5 * exp(-log_var)
            weights = {}
            for name, log_var in self.log_vars.items():
                precision = torch.exp(-log_var.detach()).item()
                weights[name] = precision
            return weights
        elif self.strategy == 'dwa':
            return self.current_weights.copy()
        else:
            return self.fixed_weights.copy()
    
    def get_uncertainties(self) -> Dict[str, float]:
        """Get learned uncertainties (only for uncertainty strategy)."""
        if self.strategy != 'uncertainty':
            return {}
        return {
            name: torch.exp(0.5 * log_var.detach()).item()  # σ = exp(0.5 * log_var)
            for name, log_var in self.log_vars.items()
        }


# Global instance for easy access from loss function
_global_loss_weighter: Optional[MultiTaskLossWeighter] = None


def get_multitask_loss_weighter() -> Optional[MultiTaskLossWeighter]:
    """Get the global loss weighter instance."""
    return _global_loss_weighter


def set_multitask_loss_weighter(weighter: MultiTaskLossWeighter):
    """Set the global loss weighter instance."""
    global _global_loss_weighter
    _global_loss_weighter = weighter


def create_multitask_loss_weighter(
    strategy: str = 'uncertainty',
    device: str = 'cuda',
    **kwargs
) -> MultiTaskLossWeighter:
    """
    Create and register a multi-task loss weighter.
    
    Args:
        strategy: 'fixed', 'uncertainty', or 'dwa'
        device: Device to create parameters on
        **kwargs: Additional arguments for the weighter
        
    Returns:
        MultiTaskLossWeighter instance
    """
    weighter = MultiTaskLossWeighter(strategy=strategy, device=device, **kwargs)
    set_multitask_loss_weighter(weighter)
    return weighter
