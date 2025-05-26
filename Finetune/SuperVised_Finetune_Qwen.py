import os
import torch
import gc
from unsloth import FastLanguageModel
from datasets import load_dataset  # Add this import
gc.collect()
torch.cuda.empty_cache()

# -----------------------------
# Paths
# -----------------------------
model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-7B-Instruct"
output_dir = os.path.join(model_path, "")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Model and Tokenizer Initialization
# -----------------------------
max_seq_length = 2048
dtype = None
load_in_4bit = True




model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map={"": "cuda:0"},
    attn_implementation="flash_attention_2"
    )



# -----------------------------
# PEFT Configuration
# -----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    
    
)

model.to("cuda:0")

# -----------------------------
# Dataset Preparation
# -----------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Format exactly as in training data
        text = alpaca_prompt.format(
            instruction.strip(),
            input_text.strip(),
            output.strip()
        ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load and format dataset
train_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\train.json"
test_file = r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\test.json"
dataset = load_dataset(
    "json",    
    data_files={
        "train": train_file,
        "test": test_file,  # Add test split
    },
    field=None
)
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
# -----------------------------
# Training Configuration
# -----------------------------
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            warmup_steps=10,
            learning_rate=2e-5,
            max_grad_norm=1.0,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            eval_strategy="steps",
            save_strategy="steps",  
            #ptim="adamw_8bit",
            eval_steps=10, 
            save_steps=10,  
            load_best_model_at_end=True,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
    print("Training complete. Stats:", trainer_stats)
    model.save_pretrained(os.path.join(output_dir, "final_model"))