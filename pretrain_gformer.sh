
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
max_qformer_length=32
lr=2e-5
wd=0.05

wandb online


dataset_dir=data/pretraining_25M_tables
num_train_epochs=3


cfg=hytrel/gformer_pretraining.cfg

include=localhost:0,1,2,3,4,5,6,7
gas=1

master_port=29511

echo ${cfg}

export WANDB_PROJECT=$(basename "$dataset_dir")

deepspeed --master_port=${master_port} --include=${include} train_sqformer.py \
    --do_train \
    --bf16 \
    --deepspeed=${deepspeed_config_file} \
    --cfg=${cfg} \
    --max_qformer_length=${max_qformer_length} \
    --dataset_dir=${dataset_dir} \
    --overwrite_output_dir \
    --output_dir=./no_ln-outputs/${dataset_dir}/${cfg} \
    --seed=0 \
    --num_train_epochs=${num_train_epochs} \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=${gas} \
    --per_device_eval_batch_size=16 \
    --save_strategy=steps \
    --save_steps=2000 \
    --save_total_limit=10 \
    --learning_rate=${lr} \
    --weight_decay=${wd} \
    --warmup_ratio=0.05 \
    --lr_scheduler_type=cosine \
    --logging_steps=10 \
    --report_to wandb