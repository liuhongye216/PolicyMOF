# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin
from .utils import per_token_loss_func, per_token_loss_func_sp
from .multitask_loss import (
    MultiTaskLossWeighter, 
    get_multitask_loss_weighter, 
    set_multitask_loss_weighter,
    create_multitask_loss_weighter
)

logger = get_logger()


class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    args: TrainingArguments

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        
        # Apply SMILES token masking if enabled (only during training)
        if self.model.training and getattr(self, 'smiles_augmenter', None) is not None:
            inputs = self._apply_smiles_token_mask(inputs)
        
        # For tasks whose `labels` are per-sample (e.g. seq_cls/reranker/embedding), we must NOT let
        # SP code treat them as token labels. We detect that case by `labels.dim() == 1` and temporarily
        # remove labels during `prepare_inputs`.
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            labels = inputs.get('labels', None)
            pop_labels = isinstance(labels, torch.Tensor) and labels.dim() == 1
            if pop_labels:
                labels = inputs.pop('labels', None)
            try:
                sequence_parallel.prepare_inputs(inputs)
            finally:
                if pop_labels and labels is not None:
                    inputs['labels'] = labels
        return inputs

    def _apply_smiles_token_mask(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SMILES token masking to input_ids.
        
        This method randomly masks a portion of SMILES tokens (excluding special tokens)
        to improve model robustness during training.
        
        Args:
            inputs: Dictionary containing 'input_ids' and optionally 'attention_mask'
            
        Returns:
            Modified inputs with masked token IDs
        """
        if 'input_ids' not in inputs:
            return inputs
        
        augmenter = self.smiles_augmenter
        if augmenter is None or not augmenter.mask_enabled:
            return inputs
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        # Apply token masking
        masked_ids, mask_positions = augmenter.apply_token_mask(input_ids, attention_mask)
        inputs['input_ids'] = masked_ids
        
        # Optionally store mask positions for analysis/logging
        inputs['_smiles_mask_positions'] = mask_positions
        
        return inputs

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            # fix device_map: handle both tensor and dict labels (for multitask)
            if isinstance(labels, dict):
                # Multitask: labels is a dict with 'cls_labels' and 'reg_labels'
                config = kwargs.get('config', None)
                pooled_logits = kwargs.get('pooled_logits', logits)
                
                cls_labels = labels.get('cls_labels')
                reg_labels = labels.get('reg_labels')
                
                num_cls_labels = getattr(config, 'num_cls_labels', None) if config else None
                num_reg_labels = getattr(config, 'num_reg_labels', 1) if config else 1
                
                if num_cls_labels is None:
                    # Fallback: infer from logits shape
                    num_cls_labels = pooled_logits.shape[-1] - num_reg_labels
                
                cls_logits = pooled_logits[..., :num_cls_labels]
                reg_logits = pooled_logits[..., num_cls_labels:num_cls_labels + num_reg_labels]
                
                cls_loss = None
                reg_loss = None
                
                if cls_labels is not None:
                    cls_labels = cls_labels.to(cls_logits.device)
                    cls_loss = nn.CrossEntropyLoss()(cls_logits.view(-1, num_cls_labels), cls_labels.view(-1))
                
                if reg_labels is not None:
                    reg_labels = reg_labels.to(reg_logits.device)
                    if num_reg_labels == 1:
                        reg_loss = nn.MSELoss()(reg_logits.squeeze(), reg_labels.squeeze())
                    else:
                        reg_loss = nn.MSELoss()(reg_logits, reg_labels)
                
                # Check for dynamic loss weighting
                loss_weighter = get_multitask_loss_weighter()
                
                if loss_weighter is not None:
                    # Use dynamic weighting (uncertainty, DWA, etc.)
                    return loss_weighter(cls_loss=cls_loss, reg_loss=reg_loss)
                else:
                    # Fallback to fixed weight
                    multitask_loss_weight = getattr(config, 'multitask_loss_weight', 1.0) if config else 1.0
                    
                    if cls_loss is not None and reg_loss is not None:
                        return cls_loss + multitask_loss_weight * reg_loss
                    elif cls_loss is not None:
                        return cls_loss
                    elif reg_loss is not None:
                        return reg_loss
                    else:
                        return torch.tensor(0.0, device=pooled_logits.device)
            else:
                labels = labels.to(logits.device)
                return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        # Register embedding dropout hook if enabled
        self._smiles_embedding_hook_handle = None
        if getattr(self, 'smiles_augmenter', None) is not None and self.smiles_augmenter.dropout_enabled:
            self._register_smiles_embedding_dropout_hook()
        
        try:
            with self._patch_loss_function():
                return super().train(*args, **kwargs)
        finally:
            # Clean up the hook
            if self._smiles_embedding_hook_handle is not None:
                self._smiles_embedding_hook_handle.remove()
                self._smiles_embedding_hook_handle = None

    def _register_smiles_embedding_dropout_hook(self):
        """Register a forward hook on the embedding layer to apply SMILES token dropout."""
        from peft import PeftModel
        
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        
        # Find the embedding layer
        embed_layer = None
        if hasattr(model, 'get_input_embeddings'):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embed_layer = model.transformer.wte
        
        if embed_layer is None:
            logger.warning('Could not find embedding layer for SMILES dropout. Dropout will be disabled.')
            return
        
        augmenter = self.smiles_augmenter
        
        def embedding_dropout_hook(module, input, output):
            """Apply dropout to embeddings after the embedding lookup."""
            if not self.model.training:
                return output
            
            # Get input_ids from the current batch (stored in _current_input_ids during compute_loss)
            input_ids = getattr(self, '_current_input_ids', None)
            attention_mask = getattr(self, '_current_attention_mask', None)
            
            if input_ids is None:
                return output
            
            # Apply embedding dropout
            return augmenter.apply_embedding_dropout(output, input_ids, attention_mask)
        
        self._smiles_embedding_hook_handle = embed_layer.register_forward_hook(embedding_dropout_hook)
        logger.info('Registered SMILES embedding dropout hook')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Store current input_ids for the embedding dropout hook
        if getattr(self, 'smiles_augmenter', None) is not None and self.smiles_augmenter.dropout_enabled:
            self._current_input_ids = inputs.get('input_ids')
            self._current_attention_mask = inputs.get('attention_mask')
        
        # Remove temporary mask positions from inputs before forward pass
        inputs.pop('_smiles_mask_positions', None)
        
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Clean up stored input_ids
        self._current_input_ids = None
        self._current_attention_mask = None
        
        return (loss, outputs) if return_outputs else loss


def gather_for_unpadded_tensors(input_data, use_gather_object=False):
    from accelerate.utils import gather_object
    from swift.trainers.sequence_parallel import sequence_parallel

    if getattr(sequence_parallel, 'dp_group', None) is not None:
        input_data = sequence_parallel._gather_object_dp(input_data)
    else:
        input_data = gather_object(input_data)
    output = []
    for _data in input_data:
        if len(_data.shape) == 0:
            _data = _data.unsqueeze(0)
        _data = _data.cpu()
        output.append(_data)
    if len(output[0].shape) == 1 and output[0].shape[0] > 1:
        data = torch.stack(output, dim=0)
    else:
        data = torch.concat(output, dim=0)
    return data


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']
        self.gather_function = gather_for_unpadded_tensors

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import calculate_paired_metrics, calculate_infonce_metrics
        args = self.args
        if args.loss_type == 'infonce':
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.include_for_metrics = ['inputs']
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        if self.args.loss_type in {'generative_reranker', 'listwise_generative_reranker'}:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None
        self.gather_function = gather_for_unpadded_tensors

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits at the last valid (non -100) timestep
        for each sample, avoiding padded timesteps created by multi-GPU gather.
        """

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last non -100 position per sample
        # Shapes: logits [batch, seq_len, vocab]
        if len(logits.shape) == 3:
            positive_logits = logits[:, :, positive_token_id]
            negative_logits = logits[:, :, negative_token_id]
            logits = positive_logits - negative_logits
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        import numpy as np
        from swift.plugin.loss import calculate_reranker_metrics
        input_ids = eval_prediction.inputs
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if self.template.padding_free:
            logits = logits[:, -1]
        else:
            if logits.ndim == 2 and logits.shape[1] > 1:
                pad_token_id = self.tokenizer.pad_token_id
                valid_mask = (input_ids != pad_token_id) & (input_ids != -100)
                last_valid_indices = valid_mask[:, ::-1].argmax(axis=1)
                last_valid_indices = input_ids.shape[1] - 1 - last_valid_indices
                logits = logits[np.arange(logits.shape[0]), last_valid_indices]
        return calculate_reranker_metrics(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if inputs.get('attention_mask') is None and self.template.padding_side != 'left':
            raise ValueError('When using padding_free, padding_side must be set to "left".')
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(
                    outputs,
                    labels,
                    num_items_in_batch=num_items_in_batch,
                    trainer=self,
                    attention_mask=inputs.get('attention_mask'))
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels, attention_mask=inputs.get('attention_mask'))

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        packing = self.template.packing
        padding_free = self.template.padding_free
        self.template.packing = False
        self.template.padding_free = False
        try:
            yield
        finally:
            self.template.packing = packing
            self.template.padding_free = padding_free
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def _prepare_inputs(self, inputs):
        from swift.llm import HfConfigFactory
        args = self.args
        inputs = super()._prepare_inputs(inputs)
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_inputs(inputs)

        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep:
            self.prepare_logits_to_keep(inputs)
            if args.tuner_backend == 'unsloth' and isinstance(inputs['logits_to_keep'], torch.Tensor):
                inputs['logits_to_keep'] = int(inputs['logits_to_keep'].sum())

        base_model = self.template.get_base_model(self.model)
        if self.model.model_info.is_moe_model and 'output_router_logits' in inspect.signature(
                base_model.forward).parameters:
            HfConfigFactory.set_config_attr(base_model.config, 'router_aux_loss_coef', args.router_aux_loss_coef)
            base_model.router_aux_loss_coef = args.router_aux_loss_coef
            logger.info_once(f'router_aux_loss_coef: {args.router_aux_loss_coef}')
            if args.router_aux_loss_coef > 0:
                inputs['output_router_logits'] = True
        inputs['compute_loss_func'] = self.compute_loss_func
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        compute_loss_func: Callable = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        text_position_ids = inputs.pop('text_position_ids', None)
        if text_position_ids is None:
            text_position_ids = inputs.get('position_ids')
        channels = inputs.pop('channel', None)

        if (self.label_smoother is not None or compute_loss_func is not None or loss_scale is not None
                or self.args.enable_dft_loss or self.args.enable_channel_loss
                or self.template.sequence_parallel_size > 1) and 'labels' in inputs:
            if self.args.use_liger_kernel:
                logger.warning_once('The cross_entropy loss function defined in Liger Kernel will not '
                                    'take effect, potentially leading to increased GPU memory consumption.')
            labels = inputs.pop('labels')
        outputs = model(**inputs)
        if getattr(outputs, 'aux_loss', None) is not None:
            mode = 'train' if self.model.training else 'eval'
            self.custom_metrics[mode]['aux_loss'].update(outputs.aux_loss)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if hasattr(self.args, 'past_index') and self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)
            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            outputs.loss = None
            if (self.args.enable_dft_loss or loss_scale is not None or self.args.enable_channel_loss
                    or self.template.sequence_parallel_size > 1):
                if self.template.sequence_parallel_size > 1:
                    outputs.loss = per_token_loss_func_sp(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)
                else:
                    outputs.loss = per_token_loss_func(outputs, labels, enable_dft_loss=self.args.enable_dft_loss)

                if loss_scale is not None:
                    loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
                    outputs.loss = outputs.loss * loss_scale

                if self.args.enable_channel_loss and channels is not None:
                    mode = 'train' if self.model.training else 'eval'
                    metrics = self.custom_metrics[mode]
                    masks = torch.roll(labels, shifts=-1, dims=-1).view(-1) != -100
                    if self.template.padding_free:
                        cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
                    else:
                        cu_seqlens = torch.arange(0, labels.shape[0] + 1) * labels.shape[1]
                    for i in range(cu_seqlens.shape[0] - 1):
                        channel = channels[i]
                        slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                        metrics[f'loss_{channel}'].update(outputs.loss[slice_][masks[slice_]])

            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)
            elif self.label_smoother is None:
                # Handle the outputs.loss generated by loss_scale.
                if num_items_in_batch is None:
                    num_items_in_batch = (labels[:, 1:] != -100).sum()
                loss = outputs.loss.sum() / num_items_in_batch
            else:
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)

            if self.model.model_info.is_moe_model and self.args.router_aux_loss_coef is not None:
                aux_loss = outputs.get('aux_loss')
                if aux_loss is not None:
                    if num_items_in_batch is not None:
                        aux_loss = aux_loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)
                    loss = loss + self.args.router_aux_loss_coef * aux_loss.to(loss.device)

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if (outputs.logits is not None and labels is not None and self.args.tuner_backend != 'unsloth'):
            cu_seqlens = None
            if self.template.padding_free and self.args.acc_strategy == 'seq':
                cu_seqlens = self.get_cu_seqlens(text_position_ids, inputs.get('logits_to_keep'))
            # Liger does not have logits
            # Unsloth has a bug with output logits
            self._compute_acc(outputs, labels, cu_seqlens=cu_seqlens)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)


class MultiTaskGenTrainer(Seq2SeqTrainer):
    """Trainer for multi-task learning that shares a backbone across
    generation (causal LM), classification, and regression tasks.

    Each sample carries a ``task`` field (``"gen"`` / ``"cls"`` / ``"reg"``).
    Generation samples use the standard next-token prediction loss;
    classification and regression samples pool the last hidden state
    and feed it through lightweight linear heads (``model.cls_head`` /
    ``model.reg_head``) attached by ``patch_multi_task_gen_heads``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ------------------------------------------------------------------
        # DDP compatibility for PEFT modules_to_save + gradient_checkpointing
        # ------------------------------------------------------------------
        # transformers auto-sets ddp_find_unused_parameters=True for PeftModel,
        # which combined with our manual cls_head/reg_head forwards (called
        # outside the DDP-wrapped model.forward) triggers the
        # "marked as ready twice" autograd error.  We force it OFF and
        # rely on the always-touch trick in compute_loss to keep every
        # head parameter receiving a gradient on every iteration.
        # ref: https://github.com/huggingface/peft/issues/899
        try:
            ddp_flag = getattr(self.args, 'ddp_find_unused_parameters', None)
            if ddp_flag is not False:
                self.args.ddp_find_unused_parameters = False
                logger.info(
                    'MultiTaskGenTrainer: forcing ddp_find_unused_parameters=False '
                    'to avoid PEFT modules_to_save + DDP "marked ready twice" error.')
        except Exception:
            pass

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        task_types = inputs.get('task')
        if task_types and any(t in ('cls', 'reg') for t in task_types):
            inputs['output_hidden_states'] = True
        return inputs

    # ------------------------------------------------------------------
    # loss helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pool_last_token(hidden_states, attention_mask, indices, device):
        """Return the hidden vector at the last non-padding position.

        Robust to both right-padding (training default) and left-padding (inference).
        """
        batch_hidden = hidden_states[indices]
        if attention_mask is not None:
            batch_mask = attention_mask[indices].long()
            seq_len = batch_mask.size(1)
            # rightmost 1 index = seq_len - 1 - argmax of flipped mask
            last_idx = seq_len - 1 - batch_mask.flip(dims=[1]).argmax(dim=1)
        else:
            last_idx = torch.full(
                (len(indices),), hidden_states.size(1) - 1,
                dtype=torch.long, device=device)
        return batch_hidden[torch.arange(len(indices), device=device), last_idx]

    def _gen_loss(self, logits, labels, gen_indices, device):
        idx = torch.tensor(gen_indices, device=device)
        shift_logits = logits[idx, :-1].contiguous()
        shift_labels = labels[idx, 1:].contiguous().to(device)
        return nn.CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    @staticmethod
    def _head_dtype(head):
        """Resolve the dtype of a possibly-wrapped (ModulesToSaveWrapper) head."""
        try:
            return next(head.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _cls_loss(self, model, hidden_states, attention_mask, cls_labels,
                  cls_indices, device):
        pooled = self._pool_last_token(
            hidden_states, attention_mask, cls_indices, device)
        base = self.accelerator.unwrap_model(model)
        head_dtype = self._head_dtype(base.cls_head)
        logits = base.cls_head(pooled.to(head_dtype))
        targets = cls_labels[cls_indices].to(device)
        loss = nn.CrossEntropyLoss()(logits.float(), targets)
        mode = 'train' if model.training else 'eval'
        preds = logits.argmax(dim=-1)
        self.custom_metrics[mode]['cls_acc'].update(
            (preds == targets).float())
        return loss

    def _reg_loss(self, model, hidden_states, attention_mask, reg_labels,
                  reg_indices, device):
        pooled = self._pool_last_token(
            hidden_states, attention_mask, reg_indices, device)
        base = self.accelerator.unwrap_model(model)
        head_dtype = self._head_dtype(base.reg_head)
        logits = base.reg_head(pooled.to(head_dtype))
        targets = reg_labels[reg_indices].to(device)
        loss = nn.MSELoss()(logits.squeeze(-1).float(), targets.float())
        mode = 'train' if model.training else 'eval'
        self.custom_metrics[mode]['reg_mse'].update(
            nn.functional.mse_loss(
                logits.squeeze(-1).float(), targets.float(), reduction='none'))
        return loss

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        task_types = inputs.pop('task', None)
        cls_labels = inputs.pop('_cls_labels', None)
        reg_labels = inputs.pop('_reg_labels', None)

        if task_types is None:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch)

        labels = inputs.pop('labels', None)
        inputs.pop('loss_scale', None)
        inputs.pop('compute_loss_func', None)
        inputs.pop('text_position_ids', None)
        inputs.pop('channel', None)

        outputs = model(**inputs)
        device = outputs.logits.device

        losses = []
        gen_indices = [i for i, t in enumerate(task_types) if t == 'gen']
        cls_indices = [i for i, t in enumerate(task_types) if t == 'cls']
        reg_indices = [i for i, t in enumerate(task_types) if t == 'reg']

        mode = 'train' if model.training else 'eval'
        if gen_indices and labels is not None:
            gl = self._gen_loss(outputs.logits, labels, gen_indices, device)
            if torch.isfinite(gl):
                losses.append(gl)
                self.custom_metrics[mode]['gen_loss'].update(gl.detach())

        hs = getattr(outputs, 'hidden_states', None)
        attn = inputs.get('attention_mask')

        if cls_indices and hs is not None and cls_labels is not None:
            cl = self._cls_loss(
                model, hs[-1], attn, cls_labels, cls_indices, device)
            if torch.isfinite(cl):
                losses.append(cl)
                self.custom_metrics[mode]['cls_loss'].update(cl.detach())

        if reg_indices and hs is not None and reg_labels is not None:
            rl = self._reg_loss(
                model, hs[-1], attn, reg_labels, reg_indices, device)
            if torch.isfinite(rl):
                losses.append(rl)
                self.custom_metrics[mode]['reg_loss'].update(rl.detach())

        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            # No usable task in this batch (or all losses NaN) – return a benign zero
            # that still depends on graph parameters so backward doesn't break.
            total_loss = outputs.logits.sum() * 0.0

        # ------------------------------------------------------------------
        # DDP "always-touch" trick.
        # With ddp_find_unused_parameters=False (which we force to avoid the
        # "marked ready twice" error from PEFT modules_to_save), every
        # trainable parameter MUST receive a gradient on every iteration on
        # every rank -- including ranks whose mini-batch happens to be 100 %
        # gen samples.  Some batches may not contain cls/reg samples and
        # `output_hidden_states` may be False, so we add a zero-contribution
        # forward through whichever head was not actually used (using a
        # fresh zero tensor that does NOT depend on hidden_states).
        # The contribution is `0 * head(zero)` so it does not affect the
        # numerical value of the loss but keeps DDP happy.
        # NOTE: must run on EVERY iteration regardless of the task mix.
        # ------------------------------------------------------------------
        if model.training:
            base = self.accelerator.unwrap_model(model)

            def _touch(head, used: bool):
                # If a head was already used in real loss, do NOT call it again
                # (would cause "marked ready twice").  Only touch unused ones.
                if used:
                    return None
                head_dtype = self._head_dtype(head)
                # head.weight is forwarded by ModulesToSaveWrapper.__getattr__
                # to the active trainable copy when the wrapper is active.
                try:
                    in_features = head.weight.shape[-1]
                except AttributeError:
                    in_features = head.modules_to_save[head.active_adapter].weight.shape[-1]
                zero_dummy = torch.zeros(1, in_features, device=device, dtype=head_dtype)
                return head(zero_dummy).sum() * 0.0

            cls_used = bool(cls_indices) and cls_labels is not None and hs is not None
            reg_used = bool(reg_indices) and reg_labels is not None and hs is not None
            extra = []
            try:
                ce = _touch(base.cls_head, cls_used)
                if ce is not None:
                    extra.append(ce)
                re_ = _touch(base.reg_head, reg_used)
                if re_ is not None:
                    extra.append(re_)
                if extra:
                    total_loss = total_loss + sum(extra)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f'MultiTaskGenTrainer always-touch failed: {e}')

        outputs.loss = total_loss
        return (total_loss, outputs) if return_outputs else total_loss
