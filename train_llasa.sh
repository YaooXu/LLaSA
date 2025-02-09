export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0

deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=3
lr=2e-5
wd=0.05
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

wandb online

dataset_dir=data/all-table-kg-schema-tasks

llm=llama


cfg=hytrel/llasa_sft_49k.cfg
master_port=29509
include=localhost:0,1,2,3,4,5,6,7
gas=4

echo ${cfg}

export WANDB_PROJECT=$(basename "$dataset_dir")

    # --gradient_checkpointing \
deepspeed --master_port=${master_port} --include=${include} train_sqformer.py \
    --do_train \
    --do_predict \
    --bf16 \
    --deepspeed=${deepspeed_config_file} \
    --cfg=${cfg} \
    --max_desc_length=${max_desc_length} \
    --max_seq_length=${max_seq_length} \
    --dataset_dir=${dataset_dir} \
    --overwrite_output_dir \
    --output_dir=./new-outputs/${dataset_dir}-10_tasks/${cfg} \
    --seed=0 \
    --num_train_epochs=${num_train_epochs} \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=${gas} \
    --per_device_eval_batch_size=16 \
    --save_strategy=steps \
    --save_steps=500 \
    --save_total_limit=1 \
    --learning_rate=${lr} \
    --weight_decay=${wd} \
    --warmup_ratio=0.05 \
    --lr_scheduler_type=cosine \
    --logging_steps=5 \
    --report_to wandb