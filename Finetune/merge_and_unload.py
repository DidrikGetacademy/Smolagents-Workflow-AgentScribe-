import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

# Paths
model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\unsloth\FineTuned Versions\phi-4-mini-instruct-Finetuned-version-1"
lora_checkpoint_dir = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\unsloth\FineTuned Versions\phi-4-mini-instruct-Finetuned-version-1\checkpoint-342"
merged_model_output_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\unsloth\FineTuned Versions\phi-4-mini-instruct-finedtuned-version-1-v2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

# Load LoRA weights
print("Loading LoRA adapters...")
lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)

# Merge and unload LoRA
print("Merging LoRA into base model...")
merged_model = lora_model.merge_and_unload()

# Save merged model
print(f"Saving merged model to {merged_model_output_path}")
merged_model.save_pretrained(merged_model_output_path)


# Cleanup
del base_model, lora_model, merged_model
gc.collect()
torch.cuda.empty_cache()

print("✅ Merge complete.")
