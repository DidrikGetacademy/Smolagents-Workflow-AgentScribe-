import os
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training

from transformers import Trainer, TrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,         # example thresholds you can tune
    llm_int8_has_fp16_weight=False  # optional
)
CONFIG = {
    "model_name": r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Ministral-8B-Instruct-2410",
    "dataset_path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Mistrail_dataset\mistrail_finetune.jsonl",
    "output_dir": r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Ministral-8B-Instruct-2410\Finjustert",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,  # slightly higher lr for LoRA
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "fp16": True,
    "max_seq_length": 4096,
    # LoRA specific params:
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
}

def preprocess(example, tokenizer):
    prompt = example['prompt']  
    completion = example['completion']

    # Tokenize prompt and completion separately (no special tokens to avoid double eos)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    completion_tokens = tokenizer(completion, add_special_tokens=False)

    input_ids = prompt_tokens['input_ids'] + completion_tokens['input_ids'] + [tokenizer.eos_token_id]

    # Create labels: mask prompt tokens with -100, keep completion tokens as is
    labels = [-100] * len(prompt_tokens['input_ids']) + completion_tokens['input_ids'] + [tokenizer.eos_token_id]

    # Pad/truncate to max_seq_length
    max_len = CONFIG['max_seq_length']
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length

    attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,

    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

    # Prepare model for int8 training (if using load_in_8bit)
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA config
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules = ["q_proj", "v_proj"],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load and preprocess dataset
    dataset = load_dataset('json', data_files=CONFIG['dataset_path'], split='train')
    print(f"Loaded {len(dataset)} examples")
    # Split dataset into train (80%) and eval (20%)
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_datasets['train']
    eval_ds = split_datasets['test']
    tokenized_train_ds = train_ds.map(lambda x: preprocess(x, tokenizer), batched=False, remove_columns=train_ds.column_names)
    tokenized_eval_ds = eval_ds.map(lambda x: preprocess(x, tokenizer), batched=False, remove_columns=eval_ds.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        num_train_epochs=CONFIG['num_train_epochs'],
        warmup_steps=CONFIG['warmup_steps'],
        fp16=CONFIG['fp16'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        logging_dir=os.path.join(CONFIG['output_dir'], "logs"),
        report_to=["tensorboard"],
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=CONFIG['save_steps'],  
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=data_collator,
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    logger.info("Saving LoRA adapters and tokenizer...")
    model.save_pretrained(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])

    logger.info(f"LoRA adapters and tokenizer saved to {CONFIG['output_dir']}")



if __name__ == "__main__":

    
    # Load dataset first to check it loads correctly
    ds = load_dataset("json", data_files={"train": CONFIG["dataset_path"]})["train"]
    print(f"🚀 Loaded dataset with {len(ds)} examples. Sample:")
    print(ds[0])  # print first example
    
    # Now run the main training function
    main()
