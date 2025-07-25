import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import  gc 
import os
from loss_logger import LossAndEvalloggingCallback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from log import log
from val_test import run_eval_comparison_test
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

def supervised_Finetune():

    log("\n----------------Loading/initalizing MODEL-------------\n")
   
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",

    )
    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct\Merged_checkpoint2316"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config, trust_remote_code=True, use_cache=False,local_files_only=True)
    model.config.use_cache = False
    print(f"Model: {model}")



    print("clearing cache after model loading...")
    gc.collect()
    torch.cuda.empty_cache()





    log("\n----------------Loading tokenizer-------------\n")


    tokenizer = AutoTokenizer.from_pretrained(model_id,local_files_only=True, trust_remote_code=True)

    

    log(f"tokenizer.pad token was None changed it to : {tokenizer.pad_token}")
    before_vocab = len(tokenizer)
    log(f"Vocab length BEFORE training: {before_vocab}")
    log(f"Tokenizer.pad_token already exist: {tokenizer.pad_token}")

    
    print("clearing cache after tokenizer...")
    gc.collect()
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.03,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )


    model_kbit = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
        )
    
    print("clearing cache after prepare model for kbit training...")
    gc.collect()
    torch.cuda.empty_cache()
        
    model = get_peft_model(model_kbit , peft_config)
    trainable = []
    frozen   = []
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)


    log(f"🟢 Trainable parameters: {trainable}")
    log(f"\n⚪️ Frozen parameters: {frozen[:5]}")
    log(f"Perft_config: {peft_config} \n ")
    log(f"Model is prepared for kbit training now...\n model: {model}\n")
    log(f"model: {model}\n")






    log("\n----------------Dataset Initizalation-------------\n")
    dataset = load_dataset(
        "json",
        data_files={
            "train": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl",
            "eval": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\test.jsonl",
            "validation": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\validation.jsonl"
        }
    )
    train_set = dataset["train"].shuffle(seed=32)
    val_data = dataset["eval"]
    validation_set = dataset["validation"]
    log(f"Training dataset size: {len(train_set)}\n")
    log(f"evaluation dataset size: {len(val_data)}\n")
    log(f"validation dataset size: {len(validation_set)}\n")


    print("clearing cache after loading dataset")
    gc.collect()
    torch.cuda.empty_cache()


    sft_output = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct\Merged_checkpoint2316"
    sft_config = SFTConfig(
            output_dir=sft_output,
            assistant_only_loss=True,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=1,
            learning_rate=1e-4,
            max_length=3100,
            #bf16=True,
            fp16=True,
            logging_steps=100,
            dataset_num_proc=4,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=150,
            eval_steps=100,
            warmup_ratio=0.1,
            save_total_limit=2,
            lr_scheduler_type="linear",
            metric_for_best_model="loss",
            chat_template_path=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct\Merged_checkpoint2316",
            report_to=None,
            dataloader_num_workers=3,
            dataloader_pin_memory=True,
            load_best_model_at_end=True,
            greater_is_better=False,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>"
        )

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_set,
            eval_dataset=val_data, 
            args=sft_config,
            callbacks=[LossAndEvalloggingCallback],
            processing_class=tokenizer,
        )
    




    Trainer_vocab_size = len(trainer.processing_class)
    model_embedding_size = trainer.model.get_input_embeddings().weight.size(0)


    if Trainer_vocab_size != model_embedding_size:
        log(f"Resizing model embeddings from {model_embedding_size} to {Trainer_vocab_size}")
        trainer.model.resize_token_embeddings(Trainer_vocab_size,mean_resizing=False)
        log(f"new_vocab_size(SFTTrainer): {Trainer_vocab_size}")
    else:
       log(f"Trengte ikke rezise, alt er likt")








   
    run_eval_comparison_test(trainer,sft_config,model,tokenizer, eval_dataset=validation_set, num_samples=5, max_new_tokens=500,phase="Before Training")
    print("clearing cache after validation test")
    gc.collect()
    torch.cuda.empty_cache()




    log(f"----------------Starting Training now----------------")
    trainer_output = trainer.train()
    print("clearing cache after training complete")


    gc.collect()
    torch.cuda.empty_cache()
    evalloss = trainer.evaluate(eval_dataset=validation_set)
    import json 
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\Evaluation_logg.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(evalloss, ensure_ascii=False, indent=2))


    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\finetuning_loss_logg.txt", "a", encoding="utf-8") as f:
        f.write("\n#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#\n")
        f.write("\n.......................FINAL FINETUNING METRICS......................................\n")
        f.write(f"✅ Trening ferdig etter {trainer_output.global_step} steg\n")
        f.write(f"📉 Slutt-trenings-loss: {trainer_output.training_loss:.4f}\n")
        f.write(f"📊 Slutt-metrics: {trainer_output.metrics}\n")




    print("clearing cache after evaluation complete")
    gc.collect()
    torch.cuda.empty_cache()

    run_eval_comparison_test(trainer,sft_config,model,tokenizer, eval_dataset=validation_set, num_samples=5, max_new_tokens=500,phase="After Training")


    log("Successfully done finetuning!")
    log("--------------Running a manual test--------------------\n\n")


    log(f"SUCCESS :D")
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    supervised_Finetune()




  









































