import easydict
from transformers.trainer_utils import PredictionOutput, speed_metrics
# from SQformer_dataset_tabert import DataCollatorForGenerating, DataCollatorForGraphSupervisedDataset
from dataset.SQformer_dataset_hytrel  import DataCollatorForGenerating, DataCollatorForGraphSupervisedDataset
import wandb
from transformers import Seq2SeqTrainer
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint
import datasets
from typing import Any, Callable, Dict, List, Tuple
import time
import pathlib
import math
import copy
from typing import TYPE_CHECKING, Optional, Union
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import speed_metrics
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.integrations.deepspeed  import is_deepspeed_zero3_enabled
from transformers.data.data_collator import DataCollator
from transformers import (
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer,
    logging,
    AdamW,
)
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from sympy import im
import pandas as pd
import numpy as np
from re import L
import os
import json
from collections import defaultdict
import sys
from transformers.utils import is_peft_available, WEIGHTS_NAME
from peft import PeftModel
from eval_json import eval_loose_json

import shutil

from utils.configure import Configure

sys.path.append("./src")


logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"


class PredictionProgressCallback(TrainerCallback):
    def __init__(self, trainer, llm_tokenizer, encoder_tokenizer, test_dataset, test_examples, num_samples=-1, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 1.
        """
        super().__init__()
        self.trainer = trainer
        self.llm_tokenizer = llm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        if num_samples == -1:
            num_samples = len(test_dataset)
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.test_examples = test_examples[:num_samples]
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        if args.should_log:
            logger.info(f"epoch {state.epoch}, {self.freq}")

        self.trainer.data_collator = DataCollatorForGenerating(
            self.llm_tokenizer, self.encoder_tokenizer)

        # generate predictions
        metrics = self.trainer.predict(self.sample_dataset, self.test_examples)

        self.trainer.data_collator = DataCollatorForGraphSupervisedDataset(
            self.llm_tokenizer, self.encoder_tokenizer)


def post_process_function(
    examples: List[Dict],
    outputs: EvalLoopOutput,
    llm_tokenizer: AutoTokenizer,
):
    # Decode the predicted tokens.
    preds = outputs.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, llm_tokenizer.pad_token_id)

    logger.info(f'begin batch decode {preds.shape}')
    generated_texts = llm_tokenizer.batch_decode(preds, skip_special_tokens=False)

    predictions = []
    for i, generated_text in enumerate(generated_texts):
        if "</s>" in generated_text:
            response = generated_text.split("</s>")[0]
        else:
            response = generated_text

        predictions.append(
            {
                "prediction": response,
            }
        )

    logger.info(f'update examples')
    for i in range(len(predictions)):
        predictions[i].update(**examples[i])

    return predictions


class StructQASeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, encoder_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.encoder_tokenizer = encoder_tokenizer
        self.llm_tokenizer = self.tokenizer

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        # for generating
        self.data_collator = DataCollatorForGenerating(
            self.llm_tokenizer, self.encoder_tokenizer)
        
        gen_kwargs = gen_kwargs.copy()

        self._gen_kwargs = gen_kwargs

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        start_time = time.time()
        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        )
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

            self.data_collator = DataCollatorForGraphSupervisedDataset(
                self.llm_tokenizer, self.encoder_tokenizer)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        metrics = output.metrics

        predictions = self.post_process_function(self.eval_examples, output, self.tokenizer)

        summary = eval_loose_json(easydict.EasyDict({}), data=predictions)
        summary['eval_avr'] = summary.pop('avr')
        output.metrics.update(summary)

        if self.args.should_log:
            cur_output_dir = f"{self.args.output_dir}/{metric_key_prefix}_{self.state.global_step}"
            os.makedirs(cur_output_dir, exist_ok=True)

            logger.info(f'writing predictions.json to {cur_output_dir}')
            with open(os.path.join(cur_output_dir, "predictions.json"), "w") as f:
                json.dump(predictions, f)

            with open(os.path.join(cur_output_dir, "predictions_summary.json"), "w") as f:
                json.dump(summary, f, indent=4)

            # Only the main node log the results by default
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ):
        self._gen_kwargs = gen_kwargs.copy()

        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        start_time = time.time()
        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        )
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        metrics = output.metrics

        predictions = self.post_process_function(predict_examples, output, self.tokenizer)

        if self.args.should_log:
            cur_output_dir = f"{self.args.output_dir}/{metric_key_prefix}_{self.state.global_step}"
            os.makedirs(cur_output_dir, exist_ok=True)

            logger.info(f'writing predictions.json to {cur_output_dir}')
            with open(os.path.join(cur_output_dir, "predictions.json"), "w") as f:
                json.dump(predictions, f)

            summary = eval_loose_json(easydict.EasyDict({}), data=predictions)
            summary['eval_avr'] = summary.pop('avr')
            output.metrics.update(summary)
            
            with open(os.path.join(cur_output_dir, "predictions_summary.json"), "w") as f:
                json.dump(summary, f, indent=4)

            # Only the main node log the results by default
            self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        # if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
        #     gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        generated_tokens = self.model.generate(**inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        # only new tokens are generated in StructQformerLLM
        gen_config = self.model.generation_config
        if (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
            )

        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _save(self, output_dir: str | None = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.model.llm and self.model.args.llm.finetuning_type != 'freeze':
            supported_classes = (PreTrainedModel,) if not is_peft_available() else (
                PreTrainedModel, PeftModel)
            # Save a trained model and configuration using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            if not isinstance(self.model.llm, supported_classes):
                _state_dict = self.model.llm.state_dict()

                if isinstance(unwrap_model(self.model.llm), supported_classes):
                    unwrap_model(self.model.llm).save_pretrained(
                        output_dir, state_dict=_state_dict, safe_serialization=self.args.save_safetensors
                    )
                else:
                    logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    torch.save(_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.llm.save_pretrained(
                    output_dir, state_dict=self.model.llm.state_dict(), safe_serialization=self.args.save_safetensors
                )

        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        output_state_dict = {
            k: _state_dict[k] for k in _state_dict if not k.startswith('llm')
        }
        
        if 'pretraining' in self.args.cfg:
            prefix = 'gformer.'
            output_state_dict = {k[len(prefix):]: v for k,v in output_state_dict.items() if k.startswith(prefix)}
            torch.save(output_state_dict, os.path.join(output_dir, "gformer.bin"))
        else:
            torch.save(output_state_dict, os.path.join(output_dir, "model.bin"))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # save cfg
        cfg_name = os.path.basename(self.args.cfg)
        Configure.save_to_file(self.model.args, os.path.join(output_dir, cfg_name))