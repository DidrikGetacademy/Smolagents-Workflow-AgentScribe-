import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import  gc
from peft import PeftModel
import os
from loss_logger import LossAndEvalloggingCallback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from neon.log import log
from val_test import run_eval_comparison_test
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"



def supervised_Finetune():

    log("\n----------------Loading/initalizing MODEL-------------\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,

    )
    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version2"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, attn_implementation="sdpa", quantization_config=bnb_config, local_files_only=True,  use_cache=False)
    model.config.use_cache = False
    log(f"Model: {model}")



    log("clearing cache after model loading...")
    gc.collect()
    torch.cuda.empty_cache()





    log("\n----------------Loading tokenizer-------------\n")


    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    tokenizer.padding_side = "right"
    tokenizer.eos_token = "<|end|>"
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    log(f"tokenizer length: {len(tokenizer)}")



    log(f"tokenizer.pad token was None changed it to : {tokenizer.pad_token}")
    before_vocab = len(tokenizer)
    log(f"Vocab length BEFORE training: {before_vocab}")
    log(f"Tokenizer.pad_token already exist: {tokenizer.pad_token}")


    log("clearing cache after tokenizer...")
    gc.collect()
    torch.cuda.empty_cache()

    peft_config_lora_Tuning = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.03,
            target_modules=["qkv_proj","o_proj","down_proj","gate_up_proj"]
    )


    model_kbit = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
        )

    log("clearing cache after prepare model for kbit training...")
    gc.collect()
    torch.cuda.empty_cache()

    peft_model = get_peft_model(model_kbit , peft_config_lora_Tuning)
    trainable = []
    frozen   = []
    for name, param in peft_model.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)


    log(f"🟢 Trainable parameters: {trainable}")
    log(f"\n⚪️ Frozen parameters: {frozen[:5]}")
    log(f"Perft_config: {peft_config_lora_Tuning} \n ")
    log(f"Model is prepared for kbit training now...\n model: {model}\n")
    log(f"model: {model}\n")






    log("\n----------------Dataset Initizalation-------------\n")
    dataset = load_dataset(
        "json",
        data_files={
            "train": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\123.jsonl",
         #   "eval": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\test.jsonl",
          # "validation": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\validation.jsonl"
        }
    )
    train_set = dataset["train"].shuffle(seed=32)
   # val_data = dataset["eval"].select(range(5))
    #validation_set = dataset["validation"]
   # log(f"Training dataset size: {len(train_set)}\n")
   # log(f"evaluation dataset size: {len(val_data)}\n")
   # log(f"validation dataset size: {len(validation_set)}\n")


    log("clearing cache after loading dataset")
    gc.collect()
    torch.cuda.empty_cache()


    sft_output = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version2"
    sft_config = SFTConfig(
            output_dir=sft_output,
            assistant_only_loss=True,
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            learning_rate = 5e-5,
            max_seq_length=7700,
            gradient_checkpointing= True,
            gradient_checkpointing_kwargs ={"use_reentrant": False},
            bf16=True,
            logging_steps=1,
            dataset_num_proc=4,
            save_strategy="epoch",
            warmup_ratio=0.05,
            max_grad_norm = 1.0,
            weight_decay=0.01,
            save_total_limit=1,
            lr_scheduler_type="cosine",
            chat_template_path=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\phi-4-mini-instruct-FinedTuned_version2",
            dataloader_num_workers=4,
            greater_is_better=False,
            pad_token="<|endoftext|>",
            eos_token="<|end|>",
            average_tokens_across_devices=False,
        )

    trainer = SFTTrainer(
            model=peft_model,
            train_dataset=train_set,
            args=sft_config,
            callbacks=[LossAndEvalloggingCallback],
            processing_class=tokenizer,
        )






   # run_eval_comparison_test(trainer,sft_config,model,tokenizer, eval_dataset=validation_set, num_samples=10, max_new_tokens=1000,phase="Before Training")
    log("clearing cache after validation test")
    # gc.collect()
    # torch.cuda.empty_cache()




    log(f"----------------Starting Training now----------------")
    trainer.train()
    log("clearing cache after training complete")

   # metrics = trainer.evaluate(eval_dataset=validation_set)
   # log(f"eval_loss: {metrics.eval_loss} \n eval_accuracy: {metrics.eval_accuracy} ")

    gc.collect()
    torch.cuda.empty_cache()
    #evalloss = trainer.evaluate(eval_dataset=validation_set)
   # import json
    # with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\Evaluation_logg.txt", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(evalloss, ensure_ascii=False, indent=2))


    # with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_loss_logg.txt", "a", encoding="utf-8") as f:
    #     f.write("\n#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#\n")
    #     f.write("\n.......................FINAL FINETUNING METRICS......................................\n")
    #     f.write(f"✅ Trening ferdig etter {trainer_output.global_step} steg\n")
    #     f.write(f"📉 Slutt-trenings-loss: {trainer_output.training_loss:.4f}\n")
    #     f.write(f"📊 Slutt-metrics: {trainer_output.metrics}\n")




    log("clearing cache after evaluation complete")
    gc.collect()
    torch.cuda.empty_cache()

    #run_eval_comparison_test(trainer,sft_config,model,tokenizer, eval_dataset=validation_set, num_samples=10, max_new_tokens=1000,phase="After Training")


    log("Successfully done finetuning!")
    log("--------------Running a manual test--------------------\n\n")


    log(f"SUCCESS :D")
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    supervised_Finetune()














































