import sys


import random
from utils.configure import Configure
import os
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PredictionOutput
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from models import LLaSA, GFormer
from dataset.SQformer_dataset_hytrel import (
    DEFAULT_GRAPH_PAD_TOKEN,
    DataCollatorForGenerating,
    DataCollatorForGraphSupervisedDataset,
    build_instruction_dataset,
)
import transformers
import torch
from typing import TYPE_CHECKING, Optional, Union
import pathlib
import logging
import json
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules import Module
from utils.LLaSATrainer import (
    StructQASeq2SeqTrainer,
    PredictionProgressCallback,
    post_process_function,
)
import wandb
import numpy as np
from collections import OrderedDict
from utils.utils import load_jsonl, print_trainable_params
import glob

logger = logging.getLogger(__name__)


@dataclass
class WarppedTrainingArguments(TrainingArguments):

    # data args
    dataset_dir: str = field(default=None)
    max_desc_length: int = field(default=2048)
    max_seq_length: int = field(default=2560)
    max_qformer_length: int = field(default=32)
    
    preprocessing_num_workers: int = field(default=8)
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    cfg: str = field(default="gformer/v3.cfg")

    output_dir: str = field(default="trainer_outputs")

    flash_attn: Optional[bool] = field(default=False)

    remove_unused_columns: int = field(default=False)

    generation_config = None

    predict_with_generate: bool = field(default=True)

    load_best_model_at_end: bool = field(default=False)
    metric_for_best_model: str = field(default="avr")
    greater_is_better: bool = field(default=True)

    disable_tqdm: bool = False

    ckpt_dir: str = field(default=None)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((WarppedTrainingArguments))
    (training_args,) = parser.parse_args_into_dataclasses()

    if training_args.ckpt_dir is not None:
        assert training_args.do_predict
        cfg_file = glob.glob(os.path.join(training_args.ckpt_dir, '*.cfg'))[0]
        logger.info(f'loading cfg file from {cfg_file}')
        
        model_args = Configure.Get_from_file(cfg_file)
        model_args.gformer.ckpt_path = None
        model_args.llm.ckpt_path = training_args.ckpt_dir
        
    else:
        model_args = Configure.Get(training_args.cfg)

    training_args.run_name = training_args.cfg

    set_seed(training_args.seed)
    torch_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.should_log else logging.ERROR,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    llm_tokenizer = AutoTokenizer.from_pretrained(model_args.llm.model_name_or_path, use_fast=False)
    if llm_tokenizer.pad_token is None:
        # for llama model
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.gformer.model_name_or_path, use_fast=False)

    hypergraph_enc_config = AutoConfig.from_pretrained(model_args.gformer.model_name_or_path)
    hypergraph_enc_config.update(
        {
            "vocab_size": len(encoder_tokenizer),
            "pre_norm": False,
            "activation_dropout": 0.1,
            "gated_proj": False,
            "llm_pad_token_id": llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id else llm_tokenizer.eos_token_id,
        }
    )

    if model_args.hytrel:
        hypergraph_enc_config.update({k:v for k,v in model_args.hytrel})

    model = LLaSA(
        model_args,
        hypergraph_enc_config,
        llm_tokenizer,
        encoder_tokenizer,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )

    if training_args.should_log:
        print_trainable_params(model)

    dataset_dir = pathlib.Path(training_args.dataset_dir)

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = build_instruction_dataset(
            dataset_dir / f"train.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            max_qformer_length=training_args.max_qformer_length,
            num_query_tokens=model_args.gformer.num_query_tokens,
            qformer_pretraining=model_args.gformer.pretraining
        )
        
        eval_dataset = build_instruction_dataset(
            dataset_dir / f"val.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            max_qformer_length=training_args.max_qformer_length,
            num_query_tokens=model_args.gformer.num_query_tokens,
            qformer_pretraining=model_args.gformer.pretraining
        )
        eval_dataset = eval_dataset.select(random.sample(range(len(eval_dataset)), k=min(1000, len(eval_dataset))))

    if training_args.do_predict:
        test_dataset = build_instruction_dataset(
            dataset_dir / f"test.pq",
            llm_tokenizer,
            encoder_tokenizer,
            max_seq_length=training_args.max_seq_length,
            max_desc_length=training_args.max_desc_length,
            max_qformer_length=training_args.max_qformer_length,
            num_query_tokens=model_args.gformer.num_query_tokens,
            training=False,
            qformer_pretraining=model_args.gformer.pretraining
        )
        test_examples = load_jsonl(dataset_dir / f"ori_test.jsonl")

        # if "debug" in training_args.output_dir:
        #     idxes = random.sample(range(len(test_dataset)), k=100)
        #     test_dataset = test_dataset.select(idxes)
        #     test_examples = [test_examples[i] for i in idxes]
    else:
        test_dataset = test_examples = None
        
    data_collator = DataCollatorForGraphSupervisedDataset(llm_tokenizer, encoder_tokenizer, model_args.gformer.pretraining)
    
    trainer = StructQASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        eval_examples=test_examples,
        tokenizer=llm_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        data_collator=data_collator,
        post_process_function=post_process_function,
        # compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        # 检查是否存在 checkpoints
        if os.path.exists(training_args.output_dir):
            checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        else:
            checkpoints = None

        # 如果存在 checkpoints，则从最新的 checkpoint 恢复训练
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            logging.info(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            latest_checkpoint = None
            logging.info("No checkpoints found. Starting a new training run.")

        trainer.train(resume_from_checkpoint=latest_checkpoint)

    if training_args.do_predict:
        trainer.data_collator = DataCollatorForGenerating(llm_tokenizer, encoder_tokenizer, False)
        logger.info("*** Predict ***")
        gen_config = {
                "do_sample": False,
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": 256,
                "pad_token_id": llm_tokenizer.eos_token_id,
            }
        metrics = trainer.predict(test_dataset, test_examples, **gen_config)
