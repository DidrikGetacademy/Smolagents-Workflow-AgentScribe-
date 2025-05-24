import os
import torch
import gc
from unsloth import FastLanguageModel
gc.collect()
torch.cuda.empty_cache()



# -----------------------------
# Paths
# -----------------------------
model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-7B-Instruct"
output_dir = os.path.join(model_path, "")
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# Model and Tokenizer initalization/loading
# -----------------------------
max_seq_length = 2048
dtype=None
Load_in_4bit=True,
fourbit_models = [
    "unsloth/Qwen2-0.5b-bnb-4bit"
]

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=Load_in_4bit
    )

# -----------------------------
#  PEFT (Parameter-Efficient Fine-Tuning)
# -----------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #It defines the dimension of the low-rank matrices used to approximate weight updates, controlling the parameter efficiency and capacity.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], #Specifies the names of the modules (layers) within the model to which the LoRA adaptation will be applied.
    lora_alpha=16, #lora_alpha is a scaling factor applied to the low-rank weights during the forward pass.It scales the updates to control their magnitude, often improving training stability.Here, a value of 16 means the LoRA updates are scaled by 16.
    lora_dropout=0,#no dropout is applied, so the LoRA layers always contribute fully during training.
    bias="none", #"none" means biases are not modified or trained during PEFT.
    use_gradient_checkpointing="unsloth", #Enables gradient checkpointing to save memory during training by recomputing parts of the graph on the backward pass.
    random_state=3407, #Sets the random seed or random state for reproducibility. Ensures that initialization and any randomness during PEFT application are deterministic.
    use_rslora=False,
    loftq_config=None, #Configuration for "loftq", which might be an additional method or module related to quantization or efficient tuning. None means no special loftq configuration is used. optional advanced feature.
)


# -----------------------------
# the Alpaca Prompt
# -----------------------------
alpaca_prompt  = """
                Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ### Instruction:
                {}
                ### Input:
                {}
                ### Response:
                {}"""
EOS_TOKEN = tokenizer.eos_token 
from datasets import load_dataset

train_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\train.jsonl"

dataset = load_dataset("json", data_files={"train": train_file}, split="train")

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)



# -----------------------------
# Set up the Training Configuration
# -----------------------------
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1, 
        warmup_steps = 20,
        max_steps = 120,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#https://medium.com/@danushidk507/qwen2-finetuning-qwen2-f89c5c9d15da

if __name__ == "__main__":
    trainer_stats = trainer.train()
    print("Done with training. Stats: ", trainer_stats)