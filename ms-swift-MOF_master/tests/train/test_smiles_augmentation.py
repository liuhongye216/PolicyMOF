# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Test for SMILES Token Augmentation

Usage:
    python -m pytest tests/train/test_smiles_augmentation.py -v
"""
import pytest
import torch
import numpy as np

from swift.llm.dataset.preprocessor.smiles_augmentation import (
    SmilesTokenAugmenter,
    SmilesEmbeddingDropoutLayer,
    create_smiles_augmenter,
    DEFAULT_SPECIAL_TOKENS,
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        self.cls_token_id = 3
        self.sep_token_id = 4
        self.all_special_ids = {0, 1, 2, 3, 4}
    
    def convert_tokens_to_ids(self, token):
        token_map = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[MASK]': 2,
            '[CLS]': 3,
            '[SEP]': 4,
        }
        return token_map.get(token, 1)  # Return UNK for unknown tokens


class TestSmilesTokenAugmenter:
    
    def test_initialization(self):
        """Test augmenter initialization."""
        augmenter = SmilesTokenAugmenter(
            mask_enabled=True,
            mask_ratio=0.15,
            dropout_enabled=True,
            dropout_ratio=0.1,
        )
        assert augmenter.mask_enabled is True
        assert augmenter.mask_ratio == 0.15
        assert augmenter.dropout_enabled is True
        assert augmenter.dropout_ratio == 0.1
    
    def test_initialize_with_tokenizer(self):
        """Test initialization with tokenizer."""
        augmenter = SmilesTokenAugmenter(mask_enabled=True)
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        assert augmenter._initialized is True
        assert augmenter.mask_token_id == 2
        assert len(augmenter._special_token_ids) > 0
    
    def test_apply_token_mask(self):
        """Test token masking functionality."""
        augmenter = SmilesTokenAugmenter(
            mask_enabled=True,
            mask_ratio=0.5,  # High ratio for testing
            seed=42,
        )
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        # Create test input: [CLS] token1 token2 token3 [SEP] [PAD]
        # IDs:                 3      5      6      7     4     0
        input_ids = torch.tensor([[3, 5, 6, 7, 4, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        
        masked_ids, mask_positions = augmenter.apply_token_mask(input_ids, attention_mask)
        
        # Special tokens should not be masked
        assert masked_ids[0, 0].item() == 3  # [CLS]
        assert masked_ids[0, 4].item() == 4  # [SEP]
        assert masked_ids[0, 5].item() == 0  # [PAD]
        
        # Some tokens should be masked
        num_masked = mask_positions.sum().item()
        assert num_masked > 0
    
    def test_apply_embedding_dropout(self):
        """Test embedding dropout functionality."""
        augmenter = SmilesTokenAugmenter(
            dropout_enabled=True,
            dropout_ratio=0.5,  # High ratio for testing
            seed=42,
        )
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        # Create test embeddings
        input_ids = torch.tensor([[3, 5, 6, 7, 4, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
        embeddings = torch.ones(1, 6, 768)
        
        dropped_embeddings = augmenter.apply_embedding_dropout(
            embeddings, input_ids, attention_mask
        )
        
        # Special token embeddings should not be dropped
        assert torch.all(dropped_embeddings[0, 0] == 1.0)  # [CLS]
        assert torch.all(dropped_embeddings[0, 4] == 1.0)  # [SEP]
        
        # Some embeddings should be dropped (zeroed)
        num_zeroed = (dropped_embeddings.sum(dim=-1) == 0).sum().item()
        assert num_zeroed > 0
    
    def test_mask_disabled(self):
        """Test that masking is skipped when disabled."""
        augmenter = SmilesTokenAugmenter(mask_enabled=False)
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        input_ids = torch.tensor([[3, 5, 6, 7, 4]])
        masked_ids, mask_positions = augmenter.apply_token_mask(input_ids)
        
        assert torch.all(masked_ids == input_ids)
        assert mask_positions.sum().item() == 0
    
    def test_dropout_disabled(self):
        """Test that dropout is skipped when disabled."""
        augmenter = SmilesTokenAugmenter(dropout_enabled=False)
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        input_ids = torch.tensor([[3, 5, 6, 7, 4]])
        embeddings = torch.ones(1, 5, 768)
        
        dropped_embeddings = augmenter.apply_embedding_dropout(embeddings, input_ids)
        
        assert torch.all(dropped_embeddings == embeddings)


class TestCreateSmilesAugmenter:
    
    def test_factory_function(self):
        """Test the factory function."""
        augmenter = create_smiles_augmenter(
            mask_enabled=True,
            mask_ratio=0.2,
            dropout_enabled=True,
            dropout_ratio=0.15,
            special_tokens=['[MOL]', '[/MOL]'],
            seed=123,
        )
        
        assert augmenter.mask_enabled is True
        assert augmenter.mask_ratio == 0.2
        assert augmenter.dropout_enabled is True
        assert augmenter.dropout_ratio == 0.15
        assert '[MOL]' in augmenter.special_tokens
        assert '[/MOL]' in augmenter.special_tokens
        # Default tokens should also be included
        assert '[CLS]' in augmenter.special_tokens


class TestSmilesEmbeddingDropoutLayer:
    
    def test_forward_training(self):
        """Test embedding dropout layer in training mode."""
        augmenter = SmilesTokenAugmenter(
            dropout_enabled=True,
            dropout_ratio=0.5,
            seed=42,
        )
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        layer = SmilesEmbeddingDropoutLayer(augmenter)
        layer.train()
        
        input_ids = torch.tensor([[3, 5, 6, 7, 4]])
        embeddings = torch.ones(1, 5, 768)
        
        output = layer(embeddings, input_ids)
        
        # Some embeddings should be dropped
        assert not torch.all(output == embeddings)
    
    def test_forward_eval(self):
        """Test embedding dropout layer in eval mode (should not drop)."""
        augmenter = SmilesTokenAugmenter(
            dropout_enabled=True,
            dropout_ratio=0.5,
            seed=42,
        )
        tokenizer = MockTokenizer()
        augmenter.initialize_with_tokenizer(tokenizer)
        
        layer = SmilesEmbeddingDropoutLayer(augmenter)
        layer.eval()
        
        input_ids = torch.tensor([[3, 5, 6, 7, 4]])
        embeddings = torch.ones(1, 5, 768)
        
        output = layer(embeddings, input_ids)
        
        # Embeddings should not be modified in eval mode
        assert torch.all(output == embeddings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
