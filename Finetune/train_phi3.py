import os
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
import logging
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback
# -----------------------------
# Paths
# -----------------------------
model_path = r"C:/Users/didri/Desktop/Programmering/VideoEnchancer program/local_model/microsoft/microsoft/Phi-3-mini-128k-instruct"
train_file = os.path.join(model_path, "train.jsonl")
test_file = os.path.join(model_path, "test.jsonl")
output_dir = os.path.join(model_path, "phi_3_finetuned")
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
#Classes
# -----------------------------

class LRSchedulerLogger(TrainerCallback):
    def __init__(self):
        self.trainer = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.trainer = kwargs.get("trainer", None)

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is not None:
            lr = self.trainer.optimizer.param_groups[0]["lr"]
            logger.info(f"📈 Learning rate at step {state.global_step}: {lr:.8f}")


class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            logger.info(f"📉 Step {state.global_step} - Loss: {logs['loss']:.4f}")
# -----------------------------
# Logging Setup
# -----------------------------
log_file = os.path.join(output_dir, "training.log")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Quantization Config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# -----------------------------
# Sample Prompt Generator
# -----------------------------
# sample_prompts = [
#     "Given the following YouTube video metadata, list the reasons why it might go viral:\n"
#     "Title: Fast Car vs Police Chase\nTags: fast, police, crash\nCategory: Entertainment\n"
#     "Views: 2M\nDuration: 3:45\nVirality Drivers:\n",

#     "Given the following YouTube video metadata, list the reasons why it might go viral:\n"
#     "Title: AI Can Now Paint Like Van Gogh\nTags: AI, art, machine learning\nCategory: Technology\n"
#     "Views: 500K\nDuration: 7:10\nVirality Drivers:\n",

#     "Given the following YouTube video metadata, list the reasons why it might go viral:\n"
#     "Title: World’s Quietest Room Test\nTags: science, experiment, sensory\nCategory: Education\n"
#     "Views: 1.2M\nDuration: 5:00\nVirality Drivers:\n"
# ]
sample_prompts = [
    "Decide whether the following text contains a motivational, inspirational, or wisdom quote.\n\n"
    "Text:\n[1.23s-9.23s] people give up right before they get what they always wanted\n\nAnswer:",

    "Decide whether the following text contains a motivational, inspirational, or wisdom quote.\n\n"
    "Text:\n[40.00s-45.00s] yeah, so the camera settings were off the whole time.\n\nAnswer:",

    "Decide whether the following text contains a motivational, inspirational, or wisdom quote.\n\n"
    "Text:\n[14.00s-20.00s] Knowing yourself is the beginning of all wisdom.\n\nAnswer:",
]

    
def run_sample_prompts(model, tokenizer, prompts, label):
    logger.info(f"🧪 Generation test: {label}")
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, use_cache=False)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nPrompt {i+1}:\n{prompt}\nGenerated:\n{decoded}\n{'='*40}")

# -----------------------------
# Load Base Model
# -----------------------------
logger.info("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

logger.info("✅ Model loaded.")
logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
logger.info(f"🧠 Total parameters: {total:,}")
logger.info(f"🛠️  Trainable parameters (LoRA): {trainable:,}")

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.unk_token  #avoids endeless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"
logger.info(f"✅ Tokenizer loaded. Pad token: {tokenizer.pad_token}")

# -----------------------------
# Generation Before Training
# -----------------------------
run_sample_prompts(model, tokenizer, sample_prompts, label="Before Fine-Tuning")

# -----------------------------
# LoRA Config
# -----------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    modules_to_save=None
)
# -----------------------------
# Dataset Preprocessing
# -----------------------------
# def format_prompt(data):
#     return (
#         "Given the following YouTube video metadata, list the reasons why it might go viral:\n"
#         f"Title: {data['title']}\n"
#         f"Tags: {', '.join(data['tags'])}\n"
#         f"Category: {data['category']}\n"
#         f"Views: {data['viewCount']}\n"
#         f"Duration: {data['duration']}\n"
#         "Virality Drivers:\n"
#     )
def format_prompt(data):
    return (
        "Decide whether the following text contains a motivational, inspirational, or wisdom quote.\n"
        "If it does, state what kind and say to save it. If not, say it is not motivational.\n\n"
        f"Text:\n{data['text']}\n\nAnswer:"
    )

def convert_to_text(example):
    input_text = format_prompt(example["input"])
    output_text = "\n".join(example["output"]) + "\n"
    return {"text": input_text + output_text}

logger.info("Loading and processing datasets...")
train_dataset = load_dataset("json", data_files=train_file, split="train").map(convert_to_text).shuffle(seed=42)
test_dataset = load_dataset("json", data_files=test_file, split="train").map(convert_to_text)
logger.info(f"✅ Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Training Config
# -----------------------------
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    learning_rate=5.0e-06,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
    lr_scheduler_type="cosine",  
    packing=True,
    max_seq_length=2048,
    
)

# -----------------------------
# Trainer Setup
# -----------------------------
logger.info("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=peft_config,
    callbacks=[LRSchedulerLogger(),LossLogger()]
)
trainer.tokenizer = tokenizer

# -----------------------------
# Train Model
# -----------------------------
count_parameters(model)
sample_input = train_dataset[0]["text"]
tokenized_sample = tokenizer(sample_input, return_tensors="pt")
logger.info(f"🔍 Sample tokenized input length: {len(tokenized_sample['input_ids'][0])}")

logger.info("🚀 Starting fine-tuning...")
train_result = trainer.train()
metrics = train_result.metrics
final_loss = metrics.get("loss")
if final_loss is not None:
    logger.info(f"✅ Final training loss: {final_loss:.4f}")
else:
    logger.warning("⚠️ No final training loss returned.")

logger.info("💾 Saving model and tokenizer...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# Save training configuration and PEFT config for reproducibility
with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
    json.dump(training_args.to_dict(), f, indent=2)

with open(os.path.join(output_dir, "peft_config.json"), "w", encoding="utf-8") as f:
    json.dump(peft_config.to_dict(), f, indent=2)


# -----------------------------
# Evaluation via trainer.evaluate
# -----------------------------
logger.info("📊 Running trainer.evaluate()...")
eval_metrics = trainer.evaluate()
for key, value in eval_metrics.items():
    if isinstance(value, float):
        logger.info(f"📉 Evaluation {key}: {value:.4f}")
    else:
        logger.info(f"📉 Evaluation {key}: {value}")

# -----------------------------
# Generation After Fine-Tuning
# -----------------------------
logger.info("🧪 Running generation test after fine-tuning...")
run_sample_prompts(trainer.model, tokenizer, sample_prompts, label="After Fine-Tuning")

# -----------------------------
# Save full test predictions
# -----------------------------
logger.info("Generating full evaluation predictions...")
eval_model = trainer.model
eval_model.eval()
test_loader = DataLoader(test_dataset, batch_size=1)
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Generating"):
        input_text = batch["text"][0]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(eval_model.device)
        output_ids = eval_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append({"input": input_text, "output": output_text})

# Save to JSON
json_path = os.path.join(output_dir, "eval_predictions.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)
logger.info(f"📝 Saved predictions to: {json_path}")

# Save to TXT (human-readable)
txt_path = os.path.join(output_dir, "eval_predictions.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for i, pred in enumerate(predictions):
        f.write(f"Sample {i+1}\nINPUT:\n{pred['input']}\nOUTPUT:\n{pred['output']}\n{'-'*50}\n")
logger.info(f"📄 Saved readable predictions to: {txt_path}")
logger.info(f"📄 Saved readable predictions to: {txt_path}")

# -----------------------------
# BLEU & ROUGE Scoring
# -----------------------------
import evaluate

logger.info("📏 Calculating BLEU and ROUGE scores...")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Extract true references and predictions (Virality Drivers section only)
references = [
    sample["text"].split("Virality Drivers:\n")[1].strip().split("\n")
    for sample in test_dataset
]
predictions_only = [
    pred["output"].split("Virality Drivers:\n")[1].strip()
    for pred in predictions
]

# Compute BLEU (expects list of references and predictions)
bleu_result = bleu.compute(predictions=predictions_only, references=references)

# Compute ROUGE (expects strings)
rouge_result = rouge.compute(
    predictions=predictions_only,
    references=["\n".join(ref) for ref in references]
)

# Log and save
logger.info(f"🧠 BLEU score: {bleu_result['bleu']:.4f}")
logger.info(f"🧠 ROUGE-L: {rouge_result['rougeL']:.4f}")

score_path = os.path.join(output_dir, "eval_scores.txt")
with open(score_path, "w", encoding="utf-8") as f:
    f.write(f"BLEU: {bleu_result['bleu']:.4f}\n")
    f.write(f"ROUGE-L: {rouge_result['rougeL']:.4f}\n")
logger.info(f"📊 Saved BLEU and ROUGE scores to: {score_path}")

# -----------------------------
# Save Best Checkpoint (Based on Eval Loss)
# -----------------------------
logger.info("💾 Checking if model should be saved as best checkpoint...")

eval_loss = eval_metrics.get("eval_loss", None)
best_ckpt_dir = os.path.join(output_dir, "best_checkpoint")
with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(eval_metrics, f, indent=2)

if eval_loss is not None:
    trainer.model.save_pretrained(best_ckpt_dir)
    tokenizer.save_pretrained(best_ckpt_dir)
    logger.info(f"🏅 Best checkpoint saved to: {best_ckpt_dir} (Eval Loss: {eval_loss:.4f})")
else:
    logger.warning("⚠️ Eval loss not found. Best checkpoint not saved.")





# -----------------------------
# Merge LoRA adapter with base model after training
# -----------------------------
import os
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def find_best_checkpoint_by_eval_loss(output_dir):
    best_checkpoint = None
    best_loss = float("inf")

    for folder in os.listdir(output_dir):
        if folder.startswith("checkpoint-"):
            log_path = os.path.join(output_dir, folder, "trainer_state.json")
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    for entry in reversed(state.get("log_history", [])):
                        if "eval_loss" in entry:
                            if entry["eval_loss"] < best_loss:
                                best_loss = entry["eval_loss"]
                                best_checkpoint = os.path.join(output_dir, folder)
                            break
    return best_checkpoint, best_loss

# Automatically find and merge the best checkpoint
best_ckpt_path, best_eval_loss = find_best_checkpoint_by_eval_loss(output_dir)
if best_ckpt_path:
    logger.info(f"🔍 Best checkpoint found: {best_ckpt_path} (Eval Loss: {best_eval_loss:.4f})")
    merged_output_path = os.path.join(model_path, "phi_3_merged")

    # Merge adapter into base model
    logger.info("🔄 Merging best checkpoint into base model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    model = PeftModel.from_pretrained(base_model, best_ckpt_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_output_path)
    tokenizer = AutoTokenizer.from_pretrained(best_ckpt_path)
    tokenizer.save_pretrained(merged_output_path)
    logger.info(f"✅ Merged model saved to: {merged_output_path}")
else:
    logger.warning("⚠️ No best checkpoint found to merge.")
