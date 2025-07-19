import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers.integrations import WandbCallback
import  gc 
import os
from loss_logger import LossAndEvalloggingCallback
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "online"  
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from log import log
from val_test import print_model_output_evaltest
  

def supervised_Finetune():

    log("\n----------------Loading/initalizing MODEL-------------\n")
   
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",attn_implementation="sdpa",  torch_dtype=torch.bfloat16, quantization_config=bnb_config, use_cache=False, local_files_only=True)


    model.config.use_cache = False
    print(f"Model: {model}")





    log("\n----------------Loading tokenizer-------------\n")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    before_vocab = len(tokenizer)
    log(f"Vocab length BEFORE training: {before_vocab}")

    log(f"Tokenizer pad_token: {tokenizer.pad_token}")


    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.02,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )


    model_kbit = prepare_model_for_kbit_training(
        model,
        )
    model = get_peft_model(model_kbit , peft_config)
    trainable = []
    frozen   = []
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)


    print("🟢 Trainable parameters:\n", "\n".join(trainable))
    print("\n⚪️ Frozen parameters:\n", "\n".join(frozen[:5]), "\n…(+ more)")
    log(f"Perft_config: {peft_config} \n ")
    log(f"Model is prepared for kbit training now...\n model: {model}\n")
    log(f"model: {model}\n")






    log("\n----------------Dataset Initizalation-------------\n")
    dataset = load_dataset(
        "json",
        data_files={
            "train": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl",
            "test": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\test.jsonl",
            "validation": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\validation.jsonl"
        }
    )
    train_set = dataset["train"].select(range(500)).shuffle(seed=32)
    test_set = dataset["test"].select(range(125))
    validation_set = dataset["validation"]
    log(f"Training dataset size: {len(train_set)}\n")
    log(f"evaluation dataset size: {len(test_set)}\n")
    log(f"validation dataset size: {len(validation_set)}\n")



    sft_output = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct"
    sft_config = SFTConfig(
            output_dir=sft_output,
            assistant_only_loss=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            max_length=2000,
            bf16=True,
            logging_steps=100,
            save_strategy="epoch",
            eval_strategy="epoch",
            warmup_ratio=0.1,
            save_total_limit=3,
            lr_scheduler_type="cosine",
            metric_for_best_model="loss",
            chat_template_path=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct",
            report_to=None,
        )

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_set,
            eval_dataset=test_set, 
            args=sft_config,
            callbacks=[LossAndEvalloggingCallback,WandbCallback],
            processing_class=tokenizer,
        )
    
    log(f"Testing Model on validationset")
    print_model_output_evaltest(trainer,sft_config,model,tokenizer, eval_dataset=validation_set, num_samples=5, max_new_tokens=500)



    log(f"----------------Starting Training now----------------")
    trainer.train()



    # if after_vocab != before_vocab:
    #     log("Tokenizer vokabular har vokst—trimmer embeddings tilbake til original")
    #     old_weights = model.get_input_embeddings().weight.data
    #     # Klipp vekk de ekstra radene:
    #     new_weights = old_weights[:before_vocab, :].clone()
    #     # Erstatt i modellen:
    #     model.get_input_embeddings().weight.data = new_weights
    #     model.config.vocab_size = before_vocab
    # else:
    #     log("Ingen ekstra tokens lagt til—ingen trimming nødvendig")




    after_training_tokenizer = len(tokenizer)
    print(len(after_training_tokenizer))
    log(f"Vocab length AFTER training: {after_training_tokenizer}")
    if after_training_tokenizer != before_vocab:
        model.resize_token_embeddings(after_training_tokenizer)
        log(f"Rezied model embedding with after training tokenizer")

    log("Successfully done finetuning!")

    log("--------------Running a manual test--------------------\n\n")
    print_model_output_evaltest(trainer,sft_config,model,tokenizer,eval_dataset=validation_set,num_samples=5,max_new_tokens=500)
    log(f"SUCCESS :D")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    supervised_Finetune()




  









































