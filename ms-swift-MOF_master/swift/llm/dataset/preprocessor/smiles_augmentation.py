# Copyright (c) Alibaba, Inc. and its affiliates.
"""
SMILES Token-level Augmentation Module

This module provides token-level augmentation techniques for SMILES (Simplified Molecular Input Line Entry System)
sequences during training. It includes:
1. Token Masking: Randomly mask SMILES tokens with a special [MASK] token
2. Token Dropout: Randomly zero out embeddings of SMILES tokens

These techniques help improve model robustness and generalization when training on molecular data.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import random
import torch
import torch.nn as nn
import numpy as np

from swift.utils import get_logger

logger = get_logger()


# Default special tokens that should NOT be masked or dropped
DEFAULT_SPECIAL_TOKENS = {
    '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
    '<s>', '</s>', '<pad>', '<unk>', '<mask>',
    '<|begin_of_text|>', '<|end_of_text|>', '<|eot_id|>',
    '<|start_header_id|>', '<|end_header_id|>',
    '<bos>', '<eos>',
    '[BOS]', '[EOS]',
}


class SmilesTokenAugmenter:
    """
    Augmenter for SMILES token sequences.
    
    Provides mask and dropout augmentation at the token level during training.
    Special tokens (like [CLS], [SEP], [PAD], etc.) are preserved and not augmented.
    
    Args:
        mask_enabled: Whether to enable token masking
        mask_ratio: Ratio of tokens to mask (0.0-1.0)
        dropout_enabled: Whether to enable token dropout (embedding zeroing)
        dropout_ratio: Ratio of token embeddings to dropout (0.0-1.0)
        special_tokens: Set of special tokens to exclude from augmentation
        mask_token_id: The token ID used for masking (if None, uses tokenizer's mask_token_id)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        mask_enabled: bool = False,
        mask_ratio: float = 0.15,
        dropout_enabled: bool = False,
        dropout_ratio: float = 0.1,
        special_tokens: Optional[Set[str]] = None,
        mask_token_id: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.mask_enabled = mask_enabled
        self.mask_ratio = mask_ratio
        self.dropout_enabled = dropout_enabled
        self.dropout_ratio = dropout_ratio
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.mask_token_id = mask_token_id
        
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()
        
        self._special_token_ids: Optional[Set[int]] = None
        self._initialized = False
        
        if mask_enabled:
            logger.info(f'SMILES token masking enabled with ratio: {mask_ratio}')
        if dropout_enabled:
            logger.info(f'SMILES token dropout enabled with ratio: {dropout_ratio}')
    
    def initialize_with_tokenizer(self, tokenizer) -> None:
        """
        Initialize special token IDs using the tokenizer.
        
        Args:
            tokenizer: The tokenizer to get special token IDs from
        """
        if self._initialized:
            return
            
        self._special_token_ids = set()
        
        # Add tokenizer's special tokens
        if hasattr(tokenizer, 'all_special_ids'):
            self._special_token_ids.update(tokenizer.all_special_ids)
        
        # Add user-specified special tokens
        for token in self.special_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    self._special_token_ids.add(token_id)
            except Exception:
                pass
        
        # Get mask token ID
        if self.mask_token_id is None:
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                self.mask_token_id = tokenizer.mask_token_id
            else:
                # Use UNK token as fallback for masking
                self.mask_token_id = tokenizer.unk_token_id
                logger.warning(
                    f'Tokenizer does not have a mask token, using UNK token (id={self.mask_token_id}) for masking.'
                )
        
        self._initialized = True
        logger.info(f'SmilesTokenAugmenter initialized with {len(self._special_token_ids)} special tokens')
    
    def _get_maskable_indices(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """
        Get indices of tokens that can be masked/dropped (excluding special tokens).
        
        Args:
            input_ids: Token IDs tensor of shape (seq_len,) or (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            List of indices that can be augmented
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]  # Take first sample for simplicity
        
        maskable_indices = []
        for idx, token_id in enumerate(input_ids.tolist()):
            # Skip special tokens
            if token_id in self._special_token_ids:
                continue
            # Skip padding (attention_mask = 0)
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask_val = attention_mask[0, idx].item()
                else:
                    mask_val = attention_mask[idx].item()
                if mask_val == 0:
                    continue
            maskable_indices.append(idx)
        
        return maskable_indices
    
    def apply_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to input tokens.
        
        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (masked_input_ids, mask_positions)
            - masked_input_ids: Input IDs with some tokens replaced by mask_token_id
            - mask_positions: Boolean tensor indicating which positions were masked
        """
        if not self.mask_enabled or not self._initialized:
            return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)
        
        batch_size = input_ids.shape[0]
        masked_input_ids = input_ids.clone()
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            sample_input_ids = input_ids[batch_idx]
            sample_attention_mask = attention_mask[batch_idx] if attention_mask is not None else None
            
            maskable_indices = self._get_maskable_indices(sample_input_ids, sample_attention_mask)
            
            if len(maskable_indices) == 0:
                continue
            
            # Randomly select indices to mask
            num_to_mask = max(1, int(len(maskable_indices) * self.mask_ratio))
            indices_to_mask = self._rng.choice(
                maskable_indices,
                size=min(num_to_mask, len(maskable_indices)),
                replace=False
            )
            
            for idx in indices_to_mask:
                if self.mask_token_id is not None:
                    masked_input_ids[batch_idx, idx] = self.mask_token_id
                    mask_positions[batch_idx, idx] = True
        
        return masked_input_ids, mask_positions
    
    def apply_embedding_dropout(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply random dropout to token embeddings.
        
        This zeros out embeddings for randomly selected non-special tokens.
        
        Args:
            embeddings: Token embeddings tensor of shape (batch, seq_len, hidden_dim)
            input_ids: Token IDs tensor for identifying special tokens
            attention_mask: Optional attention mask
            
        Returns:
            Embeddings with some token embeddings zeroed out
        """
        if not self.dropout_enabled or not self._initialized:
            return embeddings
        
        if not self.training:
            return embeddings
        
        batch_size = embeddings.shape[0]
        dropped_embeddings = embeddings.clone()
        
        for batch_idx in range(batch_size):
            sample_input_ids = input_ids[batch_idx]
            sample_attention_mask = attention_mask[batch_idx] if attention_mask is not None else None
            
            droppable_indices = self._get_maskable_indices(sample_input_ids, sample_attention_mask)
            
            if len(droppable_indices) == 0:
                continue
            
            # Randomly select indices to drop
            num_to_drop = max(1, int(len(droppable_indices) * self.dropout_ratio))
            indices_to_drop = self._rng.choice(
                droppable_indices,
                size=min(num_to_drop, len(droppable_indices)),
                replace=False
            )
            
            for idx in indices_to_drop:
                dropped_embeddings[batch_idx, idx] = 0.0
        
        return dropped_embeddings
    
    @property
    def training(self) -> bool:
        """Check if augmentation should be applied (only during training)."""
        return True  # This will be managed by the trainer
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all enabled augmentations.
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Optional attention mask
            embeddings: Optional token embeddings (for dropout)
            
        Returns:
            Dictionary with augmented tensors
        """
        result = {}
        
        if self.mask_enabled:
            masked_ids, mask_positions = self.apply_token_mask(input_ids, attention_mask)
            result['input_ids'] = masked_ids
            result['mask_positions'] = mask_positions
        else:
            result['input_ids'] = input_ids
            result['mask_positions'] = torch.zeros_like(input_ids, dtype=torch.bool)
        
        if self.dropout_enabled and embeddings is not None:
            result['inputs_embeds'] = self.apply_embedding_dropout(
                embeddings, input_ids, attention_mask
            )
        
        return result


class SmilesEmbeddingDropoutLayer(nn.Module):
    """
    A PyTorch module that applies embedding dropout for SMILES tokens.
    
    This can be inserted after the embedding layer in a model to apply
    token-level dropout during training.
    
    Args:
        augmenter: SmilesTokenAugmenter instance
    """
    
    def __init__(self, augmenter: SmilesTokenAugmenter):
        super().__init__()
        self.augmenter = augmenter
    
    def forward(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.augmenter.dropout_enabled:
            return self.augmenter.apply_embedding_dropout(
                embeddings, input_ids, attention_mask
            )
        return embeddings


def create_smiles_augmenter(
    mask_enabled: bool = False,
    mask_ratio: float = 0.15,
    dropout_enabled: bool = False,
    dropout_ratio: float = 0.1,
    special_tokens: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> SmilesTokenAugmenter:
    """
    Factory function to create a SmilesTokenAugmenter.
    
    Args:
        mask_enabled: Whether to enable token masking
        mask_ratio: Ratio of tokens to mask
        dropout_enabled: Whether to enable embedding dropout
        dropout_ratio: Ratio of embeddings to dropout
        special_tokens: List of special tokens to exclude
        seed: Random seed
        
    Returns:
        Configured SmilesTokenAugmenter instance
    """
    special_token_set = None
    if special_tokens is not None:
        special_token_set = set(special_tokens) | DEFAULT_SPECIAL_TOKENS
    
    return SmilesTokenAugmenter(
        mask_enabled=mask_enabled,
        mask_ratio=mask_ratio,
        dropout_enabled=dropout_enabled,
        dropout_ratio=dropout_ratio,
        special_tokens=special_token_set,
        seed=seed,
    )
