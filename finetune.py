from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
from torch.utils.data import DataLoader
import os


@dataclass
class FinetuneArguments:
    data_path: str = field(default="tatsu-lab/alpaca")
    is_data_local: bool = field(default=False)
    model_path: str = field(default="THUDM/chatglm3-6b")
    org_model: str =  field(default="THUDM/chatglm3-6b")
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    use_lora: bool = field(default=False)
    trust_remote_code: bool = field(default=True)

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss
        
    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") 
            for k, v in self.model.named_parameters() 
            if v.requires_grad
        }
        
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    """
    # Arguments is parsed by HfArgumentParser
    # About 90 arguments in TrainingArguments, the important ones in training and finetune is below:
       * output_dir
       * num_train_epochs
       * lr_scheduler_type
       * load_best_model_at_end
       * metric_for_best_model
       * greater_is_better
       * optim
       * group_by_length
       * length_column_name
       * save_steps
       * eval_steps
       * learning_rate
       
    If you want to config other ones, please check the source code in transformers
    """

    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (
            FinetuneArguments, 
            TrainingArguments
        )
    ).parse_args_into_dataclasses()

    # init model
    model = AutoModel.from_pretrained(
        finetune_args.model_path,
        load_in_8bit=True, 
        trust_remote_code=finetune_args.trust_remote_code,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(finetune_args.org_model, trust_remote_code=finetune_args.trust_remote_code)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False # silence the warnings. Please re-enable for inference!
    )
    
    print("finetune_args.use_lora",finetune_args.use_lora)
    print("training_args.fp16",training_args.fp16)
    
    if finetune_args.use_lora:
        # setup peft
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=finetune_args.lora_alpha,
            lora_dropout=finetune_args.lora_dropout,
        )
        
        model = get_peft_model(model, peft_config)
        
    # load dataset
    if finetune_args.is_data_local:
        dataset = datasets.load_from_disk(finetune_args.data_path)
    else:
        dataset = datasets.load_dataset(finetune_args.data_path)

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    
    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

