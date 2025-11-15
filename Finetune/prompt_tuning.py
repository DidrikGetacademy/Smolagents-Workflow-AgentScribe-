import os
import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, prepare_model_for_kbit_training, PromptTuningConfig, PromptTuningInit, TaskType
from neon.log import log


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
printed_example = False
def supervised_finetune():
    log("\n----------------Loading/initializing MODEL-------------\n")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-Instruct-finetuned-Motivational-text"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        local_files_only=True,
        use_cache=False
    )
    model.config.use_cache = False
    log(f"Model: {model}")


    gc.collect()
    torch.cuda.empty_cache()

    log("\n----------------Dataset Initialization-------------\n")
    dataset = load_dataset(
        "json",
        data_files={
            "train": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl",
        }
    )
    train_set = dataset["train"].shuffle(seed=32)

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"


    tokenizer.padding_side="right"

    tokenizer.eos_token = "<|end|>"
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    log(f"Tokenizer length: {len(tokenizer)}")
    log(f"Tokenizer.pad_token: {tokenizer.pad_token}")
    before_vocab = len(tokenizer)
    log(f"Vocab length BEFORE training: {before_vocab}")


    gc.collect()
    torch.cuda.empty_cache()





    def preprocess_function(examples):
        global printed_example
        messages = examples["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ) + tokenizer.eos_token

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=6144,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        labels = input_ids.clone()


        assistant_index = text.rfind("<|assistant|>")
        if assistant_index == -1:
            labels = torch.full_like(labels, -100)
        else:

            prefix_text = text[:assistant_index + len("<|assistant|>")]
            mask_until = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])

            labels[:mask_until] = -100


        if not printed_example:

                log("FULL TEXT: " + text)


                log("INPUT_IDS: " + repr(input_ids.tolist()))


                log("LABELS: " + repr(labels.tolist()))


                log("Masked tokens: " + str((labels == -100).sum().item()))


                non_masked_start = (labels != -100).nonzero(as_tuple=True)[0][0].item() if (labels != -100).any() else -1
                if non_masked_start != -1:
                    loss_text = tokenizer.decode(input_ids[non_masked_start:])
                    log("Her er teksten som blir kalkulert loss på (LOSS TEXT): " + loss_text)
                else:
                    log("No loss text (all masked)")

                log("-" * 80)

                printed_example = True

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist()
        }



    train_dataset = train_set.map(
        preprocess_function,
        remove_columns=train_set.column_names
    )


    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=50,
        prompt_tuning_init_text="Extract self-contained motivational texts verbatim from transcripts, including quotes, advice, inspiring messages, or mindset facts that promote growth, resilience, discipline, or perseverance: Ensure completeness by evaluating each text as a full, coherent thought with no unresolved references, abrupt endings, or missing elements that would confuse a standalone listener; confirm it does not lack context by verifying it inspires independently without needing surrounding explanation. Save only qualifying passages as-is, with timestamps, if they are concise and evoke positive change.",
        tokenizer_name_or_path=model_id
    )


    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )


    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version3",
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        num_train_epochs=2,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_total_limit=2,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        dataloader_num_workers=6,
        report_to="none",
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
        resume_from_checkpoint=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version3"
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    # Start training
    log("\n----------------Starting Training-------------\n")
    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

    # Save the fine-tuned model and adapter
    trainer.save_model(r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-Instruct-finetuned-Motivational-text\prompt_tuned_2")
    tokenizer.save_pretrained(r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-Instruct-finetuned-Motivational-text\prompt_tuned_2")

    log("\n----------------Training Complete-------------\n")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    supervised_finetune()
