import os
import warnings
from accelerate import infer_auto_device_map, init_empty_weights
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from peft import PeftModel
os.environ["WANDB_MODE"] = "online"  
import  gc 
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Trainer
from math import exp
from datetime import datetime


log_finetune_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\finetuning_general_log_finetune.txt"
log_finetune_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\Manual_test_output.txt"
log_finetuneged_messages = set()


def log_finetune(msg):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_finetuneged_messages.add(msg)
    log_finetune_message = f"[{timestamp}] {msg}"
    with open(log_finetune_file_path, "a", encoding="utf-8") as f:
            f.write(log_finetune_message + "\n")



def log_finetune_test(msg):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_finetuneged_messages.add(msg)
    log_finetune_message = f"[{timestamp}] {msg}"
    with open(log_finetune_file_path, "a", encoding="utf-8") as f:
            f.write(log_finetune_message + "\n")



def print_model_output_evaltest(trainer, sft_config, model, tokenizer, eval_dataset, num_samples=3, max_new_tokens=150):
    model.eval()
    log_finetune_test("Running an eval test after training to see model's performance...\n")

    # 1) Forbered datasettet
    prepared_eval_dataset = trainer._prepare_dataset(
        dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
        packing=False,
        formatting_func=None,
        dataset_name="eval",
    )

    # 2) Loop over eksempler
    for i in range(min(num_samples, len(prepared_eval_dataset))):
        example = prepared_eval_dataset[i]
        input_ids_list = example["input_ids"]
        assistant_mask = example.get("assistant_masks", None)

        # 3) Finn split‑punkt: første assistant-token (mask==1)
        if assistant_mask is not None:
            first_assistant_idx = next((idx for idx, m in enumerate(assistant_mask) if m == 1), len(input_ids_list))
        else:
            first_assistant_idx = len(input_ids_list)

        # 4) Klipp bort alt fra assistant‑start => kun bruker‑prompt
        user_input_ids = input_ids_list[:first_assistant_idx] #For 7b mistarl modell til og fungere


        # --- NYTT: log_finetune rå‑tekst av bruker‑prompten ---
        raw_input_text = tokenizer.decode(user_input_ids, skip_special_tokens=False)
        log_finetune_test(f"Raw input text for example {i+1}: {raw_input_text}")

        # 5) Konverter til tensor med batch‑dim
        input_ids = torch.tensor(user_input_ids, device=model.device).unsqueeze(0)

        # 6) Generer svar
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 7) Dekode ground truth assistant (for log_finetuneging)
        if assistant_mask is not None:
            gt_assistant_ids = input_ids_list[first_assistant_idx:]
            decoded_assistant = tokenizer.decode(gt_assistant_ids, skip_special_tokens=True)
        else:
            decoded_assistant = "<No assistant response in example>"

        # 8) Dekode modellen sin generering (kun ny generert tekst)
        gen_ids = output_ids[0][input_ids.shape[-1]:]
        model_generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 9) Print og log_finetunege
        detailed_log_finetune = (
            f"\n🔹 Example {i+1}\n"
            f"📥 Raw model input:\n{raw_input_text}\n\n"
            f"🎯 Ground truth assistant response:\n{decoded_assistant}\n\n"
            f"🧠 Model generated response:\n{model_generated_text}\n"
            + ("-" * 60)
        )
        log_finetune_test(detailed_log_finetune)
  





def print_model_output_evaltest_mistral(trainer, sft_config, model, tokenizer, eval_dataset, num_samples=3, max_new_tokens=150):
    model.eval()
    print("Running an eval test after training to see model's performance...\n")

    # 1) Forbered datasettet
    prepared_eval_dataset = trainer._prepare_dataset(
        dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
        packing=False,
        formatting_func=None,
        dataset_name="eval",
    )

    # 2) Loop over eksempler
    for i in range(min(num_samples, len(prepared_eval_dataset))):
        example = prepared_eval_dataset[i]
        input_ids_list = example["input_ids"]
        assistant_mask = example.get("assistant_masks", None)
        if assistant_mask is None:
            print(f"Assistant mask er None")

        # 3) Finn split‑punkt: første assistant-token (mask==1)
        if assistant_mask is not None:
            # Finn split‑punkt (første assistent‐token)
            first_assistant_idx = next((i for i,m in enumerate(assistant_mask) if m==1), len(input_ids_list))

            # Ta med én ekstra ID slik at prompt ender med akkurat én assistent‑token
            user_input_ids = input_ids_list[: first_assistant_idx + 2]
            print(f"assistant mask is not none")


        # --- NYTT: log_finetune rå‑tekst av bruker‑prompten ---
        raw_input_text = tokenizer.decode(user_input_ids, skip_special_tokens=False)
        log_finetune_test(f"Raw input text for example {i+1}: {raw_input_text}")


        # 5) Konverter til tensor med batch‑dim
        input_ids = torch.tensor(user_input_ids, device=model.device).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(user_input_ids), device=model.device).unsqueeze(0)
        
        # 6) Generer svar
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
    
            )

 
        if assistant_mask is not None:
            gt_assistant_ids = input_ids_list[first_assistant_idx:]
            decoded_assistant = tokenizer.decode(gt_assistant_ids, skip_special_tokens=False)
        else:
            decoded_assistant = "<No assistant response in example>"


        gen_ids = output_ids[0][ input_ids.shape[-1] : ]
        model_generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        detailed_log_finetune = (
            f"\n🔹 Example {i+1}\n"
            f"📥 Raw model input:\n{raw_input_text}\n\n"
            f"🎯 Ground truth assistant response:\n{decoded_assistant}\n\n"
            f"🧠 Model generated response:\n{model_generated_text}\n"
            + ("-" * 60)
        )
        print(detailed_log_finetune)
        log_finetune_test(detailed_log_finetune)




 
class LossAndEvallog_finetunegingCallback(TrainerCallback):
    def __init__(self):
        self.prev_train_loss = None

    def _write_train_log_finetune(self, args, state, log_finetunes):
        loss = log_finetunes.get("loss")
        grad_norm = log_finetunes.get("grad_norm", "N/A")
        lr = log_finetunes.get("learning_rate", "N/A")
        num_tokens = log_finetunes.get("num_tokens", "N/A")
        token_acc = log_finetunes.get("mean_token_accuracy", "N/A")
        step = state.global_step
        epoch = state.epoch or 0.0
        samples_seen = args.per_device_train_batch_size * args.gradient_accumulation_steps * step
        perplexity = exp(loss) if loss is not None else "N/A"

    
        if loss is not None:
            change = loss - self.prev_train_loss if self.prev_train_loss is not None else None
            if change is None:
                comment = "🚀 Starter opp!\nFørste loss – bruker som referanse fremover."
                color = "\033[94m"
            elif change < -0.1:
                comment = f"📉 Klar forbedring!\nLoss sank med {change:.4f} siden sist. God fremgang!"
                color = "\033[92m"
            elif change < -0.01:
                comment = f"✅ Litt bedre!\nLoss ned med {change:.4f}. Stabil læring."
                color = "\033[92m"
            elif abs(change) < 0.001:
                comment = f"😐 Står omtrent stille... ({change:.4f})\nMulig metningspunkt eller behov for justering."
                color = "\033[93m"
            elif change > 0.1:
                comment = f"⚠️ Loss øker mye! ({change:.4f})\nMulig overtrening, datastøy eller for høy læringsrate."
                color = "\033[91m"
            else:
                comment = f"🔄 Litt ustabilt... ({change:.4f})\nIkke alvorlig, men følg med på trenden."
                color = "\033[93m"
            self.prev_train_loss = loss
        else:
            comment = "ℹ️ Ingen loss log_finetuneget dette steget."
            color = "\033[90m"

        log_finetune_msg = (
            f"[----------------TRAINING METRICS--------------]"
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] TRAIN | Step: {step} | Epoch: {epoch:.2f} | Samples: {samples_seen}\n"
            f"🔹 Loss: {f'{loss:.4f}' if loss is not None else 'N/A'} | Perplexity: {perplexity}\n"
            f"🔹 LR: {lr}\n"
            f"🔹 Grad Norm: {grad_norm}\n"
            f"🔹 #Tokens: {num_tokens}\n"
            f"🔹 Token Acc: {token_acc}\n\n"
            f"{comment}\n"
            + "-"*60
            + "\n\n\n"
        )
        print(log_finetune_msg)
        with open(log_finetune_file_path, "a", encoding="utf-8") as f:
            f.write(log_finetune_msg + "\n")

    def on_log_finetune(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, log_finetunes=None, **kwargs):
        # Kjør trenings‑log_finetuneging når loss finnes
        if log_finetunes and log_finetunes.get("loss") is not None:
            self._write_train_log_finetune(args, state, log_finetunes)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        step = state.global_step
        epoch = state.epoch or 0.0
        eval_perplexity = exp(eval_loss) if eval_loss is not None else "N/A"

        comment = "ℹ️ Evaluerings‑runde fullført."
        log_finetune_msg = (
            f"[----------------EVALUATION METRICS--------------]"
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  EVAL  | Step: {step} | Epoch: {epoch:.2f}\n"
            f"🔹 Eval Loss: {f'{eval_loss:.4f}' if eval_loss is not None else 'N/A'} | Perplexity: {eval_perplexity}\n"
            f"{comment}\n"
            + "-"*60
            + "\n\n\n"
        )
        print(log_finetune_msg)
        with open(log_finetune_file_path, "a", encoding="utf-8") as f:
            f.write(log_finetune_msg + "\n")




def main():
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
     

    )
    log_finetune("\n----------------Loading/initalizing MODEL-------------\n")
    model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-3B-Instruct-v0.2"
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, offload_folder="offload", cache_dir=False)
    log_finetune(f"Model: {model}")
    print(f"Model: {model}")




    log_finetune("\n----------------Loading tokenizer-------------\n")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log_finetune(f"Tokenizer pad_token set to EOS: {tokenizer.eos_token}")
    else:
        log_finetune(f"Tokenizer pad_token already set: {tokenizer.pad_token}")



    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    log_finetune("\n----------------PerftConfig/LoraConfig-------------\n")
    log_finetune(f"{peft_config}")




    log_finetune("\n----------------Preparing model for kbit training-------------\n")
    model = prepare_model_for_kbit_training(model)
    log_finetune(f"Model is prepared for kbit training now...\n model: {model}\n")




    log_finetune("\n----------------get_peft_model-------------\n")
    model = get_peft_model(model,peft_config)
    log_finetune(f"model: {model}")



    log_finetune("\n----------------Dataset Initizalation-------------\n")
    dataset = load_dataset(
        "json",
        data_files={
            "train": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\train.jsonl",
            "test": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\test.jsonl",
            "validation": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\validation.jsonl"
        }
    )

    train_set = dataset["train"].shuffle(seed=25)
    test_set = dataset["test"]
    validation_set = dataset["validation"]

    log_finetune(f"Training dataset size: {len(train_set)}\n")
    log_finetune(f"evaluation dataset size: {len(test_set)}\n")
    log_finetune(f"validation dataset size: {len(validation_set)}\n")



    sft_output = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-3B-Instruct-v0.2\FineTuned"
    sft_config = SFTConfig(
            output_dir=sft_output,
            run_name="sexydidrik", 
            assistant_only_loss=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            max_length=2700,
            bf16=True,
            log_finetuneging_steps=20,
            eval_strategy="steps",
            save_steps=60,
            eval_steps=30, 
            save_total_limit=2,
            warmup_steps=25,
            lr_scheduler_type="linear",
            load_best_model_at_end=True,  
            metric_for_best_model="loss",
            chat_template_path=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-3B-Instruct-v0.2",
            dataloader_num_workers=6,
            report_to="wandb"
        )

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_set,
            eval_dataset=test_set, 
            args=sft_config,
            callbacks=[LossAndEvallog_finetunegingCallback,WandbCallback],
            processing_class=tokenizer
        )
    
    print_model_output_evaltest_mistral(trainer,sft_config,model,tokenizer,eval_dataset=validation_set,num_samples=3,max_new_tokens=256)
    log_finetune(f"EVAL DONEEEEEEE!!!!!!!!!!!!!!!!!!!!!!")
    import time
    time.sleep(20)

    
    log_finetune(f"----------------Starting Training now----------------")
    trainer.train()
    log_finetune(f"----------------✅TRAINING COMPLETE✅----------------")
    tokenizer.save_pretrained(sft_config.output_dir)
    model.save_pretrained(sft_config.output_dir)
    log_finetune(f"✅ LoRA adapter saved to: {sft_config.output_dir}")



    log_finetune("--------------Running a manual test--------------------\n\n")
    print_model_output_evaltest_mistral(trainer,sft_config,model,tokenizer,eval_dataset=validation_set,num_samples=3,max_new_tokens=256)
    log_finetune(f"EVERYTHING DONE")




if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()
