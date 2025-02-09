export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# deepspeed_config_file=ds_zero2_offload.json
deepspeed_config_file=ds_zero2.json
max_desc_length=2048
max_seq_length=2560
num_train_epochs=3
lr=2e-5
wd=0.05
master_port=29501
strategy=v2.6
num_query_tokens=10
cross_attention_freq=1
finetuning_type=freeze_backbone

wandb offline

dataset_dir=data/hytrel/all-table-kg-schema-tasks

llm=llama

gas=8

# cfg=hytrel-mistral/v2-predict1.cfg
# include=localhost:0,1,2,3,4,5,6,7
# master_port=29500

# ckpt_dir=outputs/data/hytrel/all-table-kg-schema-tasks-w_inverse_rel/hytrel-codellama/v2-10M_query_token_200k-only_query-not_freeze_gnn-trained_roberta_base-lora4_32_llama-10.cfg/checkpoint-11469
ckpt_dir=new-outputs-old/data/hytrel/all-table-kg-schema-tasks-10_tasks/hytrel-llama3/v2-lora4_32-llama-0.cfg/checkpoint-88176

include=localhost:0,1,2,3,4,5,6,7
master_port=29501

echo ${ckpt_dir}

export WANDB_PROJECT=$(basename "$dataset_dir")

    # --gradient_checkpointing \
deepspeed --master_port=${master_port} --include=${include} train_sqformer.py \
    --do_predict \
    --bf16 \
    --ckpt_dir=${ckpt_dir} \
    --output_dir=$(dirname "$ckpt_dir") \
    --max_desc_length=${max_desc_length} \
    --max_seq_length=${max_seq_length} \
    --dataset_dir=${dataset_dir} \
    --overwrite_output_dir \
    --seed=0 \
    --num_train_epochs=${num_train_epochs} \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=${gas} \
    --per_device_eval_batch_size=16 \
    --save_strategy=epoch \
    --evaluation_strategy=epoch \
    --save_total_limit=3 \
    --learning_rate=${lr} \
    --weight_decay=${wd} \
    --warmup_ratio=0.05 \
    --lr_scheduler_type=cosine \
    --logging_steps=2