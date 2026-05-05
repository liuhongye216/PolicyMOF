# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional

import numpy as np

from .core import ResponsePreprocessor, RowPreprocessor


class GroundingMixin:
    """This class offers prompts to the grounding task"""
    task_type: Optional[str] = None

    _grounding_language_mixin = [0.8, 0.2]
    _grounding_prompts = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'), ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')]
        },
        'caption': {
            'en': [
                ('<bbox>', '<ref-object>'),
                ('The object at position <bbox>', '<ref-object>'),
                ('This <bbox> is', '<ref-object>'),
                ('What is the object at <bbox>', '<ref-object>'),
                ('Describe <bbox>', '<ref-object>'),
                ('<bbox> is', '<ref-object>'),
                ('The bounding box coordinate <bbox> contains', '<ref-object>'),
            ],
            'zh': [
                ('<bbox>', '<ref-object>'),
                ('<bbox>是什么', '<ref-object>'),
                ('<bbox>的位置包含', '<ref-object>'),
                ('描述<bbox>', '<ref-object>'),
                ('<bbox>中是', '<ref-object>'),
                ('坐标<bbox>描述了什么', '<ref-object>'),
                ('描述<bbox>中的事物', '<ref-object>'),
            ]
        },
    }

    def construct_grounding_prompt(self):
        # TODO Only support one bbox to one object
        lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
        prompts = GroundingMixin._grounding_prompts[self.task_type][lang]
        query, response = prompts[np.random.choice(range(len(prompts)))]
        return query, response


class TextGenerationPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 *,
                 prompt: str,
                 query_tag: str = '{{QUERY}}',
                 columns: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.query_tag = query_tag
        self.prompt = prompt
        super().__init__(columns=columns, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = self.prompt.replace(self.query_tag, row['query'])
        return super().preprocess(row)


class ClsGenerationPreprocessor(ResponsePreprocessor):

    def __init__(self,
                 labels: List[str],
                 *,
                 task: str,
                 is_pair_seq: bool = False,
                 columns: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        self.labels = labels
        self.task = task
        self.is_pair_seq = is_pair_seq

        category = ', '.join(labels)
        self.sentence2_key = 'sentence2'
        self.label_key = 'label'
        if is_pair_seq:
            self.sentence_key = 'sentence1'
            inputs = 'Sentence1: {sentence1}\nSentence2: {sentence2}'
        else:
            self.sentence_key = 'sentence'
            inputs = 'Sentence: {sentence}'
        self.prompt = f"""Task: {task}
{inputs}
Category: {category}
Output:"""
        super().__init__(columns=columns, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        label = row.pop(self.label_key, None)
        if label is None:
            return

        if self.is_pair_seq:
            query = self.prompt.format(sentence1=row.pop(self.sentence_key), sentence2=row.pop(self.sentence2_key))
        else:
            query = self.prompt.format(sentence=row.pop(self.sentence_key))
        row['query'] = query
        row['response'] = self.labels[int(label)]
        return super().preprocess(row)


class IsomericSmilesPreprocessor(RowPreprocessor):
    """Preprocessor for SMILES augmentation in molecular data.
    
    This preprocessor supports switching between canonical and isomeric (randomized) SMILES
    representations during training to improve model robustness.
    
    Dataset format (single isomeric SMILES):
        {
            "messages": [{"role": "user", "content": "canonical_smiles ..."}],
            "isomeric_smiles": "randomized_smiles ...",
            "label": {"cls": 1, "reg": 0.755}
        }
    
    Dataset format (multiple isomeric SMILES - recommended for better augmentation):
        {
            "messages": [{"role": "user", "content": "canonical_smiles ..."}],
            "isomeric_smiles": ["randomized_smiles_1 ...", "randomized_smiles_2 ...", ...],
            "label": {"cls": 1, "reg": 0.755}
        }
    
    When use_isomeric_smiles=True, the content in messages will be replaced with
    a randomly selected isomeric SMILES from the list (or the single string if not a list).
    """

    def __init__(
            self,
            *,
            use_isomeric_smiles: bool = False,
            isomeric_field: str = 'isomeric_smiles',
            columns: Optional[Dict[str, str]] = None,
            random_state: Optional[np.random.RandomState] = None,
            **kwargs) -> None:
        """Initialize the IsomericSmilesPreprocessor.
        
        Args:
            use_isomeric_smiles: Whether to use isomeric SMILES instead of canonical.
            isomeric_field: The field name containing isomeric SMILES in the dataset.
                           Can be a string or a list of strings.
            columns: Column mapping dictionary.
            random_state: Random state for reproducible random selection.
            **kwargs: Additional arguments passed to parent class.
        """
        self.use_isomeric_smiles = use_isomeric_smiles
        self.isomeric_field = isomeric_field
        if random_state is None:
            random_state = np.random.RandomState()
        self._isomeric_random_state = random_state
        super().__init__(columns=columns, random_state=random_state, **kwargs)

    def _select_random_smiles(self, isomeric_smiles):
        """Select a random SMILES from the list or return the string directly.
        
        Args:
            isomeric_smiles: Either a string or a list of strings.
            
        Returns:
            A single SMILES string.
        """
        if isinstance(isomeric_smiles, list):
            if len(isomeric_smiles) == 0:
                return None
            # Randomly select one from the list
            idx = self._isomeric_random_state.randint(0, len(isomeric_smiles))
            return isomeric_smiles[idx]
        else:
            # Single string, return as is
            return isomeric_smiles

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single row, optionally replacing canonical SMILES with isomeric SMILES.
        
        If isomeric_smiles is a list, randomly selects one for training data augmentation.
        
        Args:
            row: A dictionary containing the data row.
            
        Returns:
            The preprocessed row, or None if the row should be skipped.
        """
        if self.use_isomeric_smiles and self.isomeric_field in row:
            isomeric_smiles_data = row.pop(self.isomeric_field)
            if isomeric_smiles_data is not None and 'messages' in row:
                # Select a random SMILES if it's a list
                selected_smiles = self._select_random_smiles(isomeric_smiles_data)
                if selected_smiles is not None:
                    messages = row['messages']
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        # Replace the content in the first user message with selected isomeric SMILES
                        for msg in messages:
                            if msg.get('role') == 'user':
                                msg['content'] = selected_smiles
                                break
        elif self.isomeric_field in row:
            # Remove the isomeric_smiles field if not using it
            row.pop(self.isomeric_field, None)
        
        # Return the row directly without calling parent's preprocess
        # since we only need to modify the messages content
        return row

