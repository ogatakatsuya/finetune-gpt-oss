import os

import wandb
import weave
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import LoraConfig, get_peft_model

os.environ["WANDB_PROJECT"] = "finetune-gpt-oss"
wandb.init(project=os.environ["WANDB_PROJECT"])
weave.init(os.environ["WANDB_PROJECT"])

model_name = "openai/gpt-oss-20b"
dataset = load_dataset("Sakaji-Lab/JaFIn", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def formatting_func(data):
    messages = [
        {'role': 'user', 'content': data['instruction']},
        {'role': 'assistant', 'content': data['output']},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

quant_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    use_cache=False,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()


training_args = SFTConfig(
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-finetuned-JaFIn",
    report_to="wandb",
    push_to_hub=False,
)


trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
    processing_class=tokenizer,
)
trainer.train()

trainer.save_model(training_args.output_dir)
