This is a lora finetune program which is based on peft.
<br>

Running like below

```python
python finetune.py --data_path tatsu-lab/alpaca \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--num_train_epochs 16 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_steps 1000 \
--eval_steps 1000 \
--save_total_limit 2 \
--learning_rate 3e-4 \
--remove_unused_columns False \
--warmup_steps 10 \
--logging_steps 10 \
--group_by_length True \
--output_dir trained_models/chatglm_lora \
--use_lora True \
--is_data_local False \
--trust_remote_code True
```
