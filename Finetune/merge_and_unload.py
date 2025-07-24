import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

# Paths
model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct"
lora_checkpoint_dir       = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct\checkpoint-2316"
merged_model_output_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct\Merged_checkpoint2316"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = "<|endoftext|>"

# Load base model in full bfloat16 precision
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

# Resize model embeddings to match tokenizer
if len(tokenizer) != base_model.get_input_embeddings().weight.size(0):
    print(f"Resizing model embeddings from {base_model.get_input_embeddings().weight.size(0)} to {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))

# Load LoRA weights
print("Loading LoRA adapters...")
lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir,is_trainable=False)

# Merge and unload LoRA
print("Merging LoRA into base model...")
merged_model = lora_model.merge_and_unload()

# Save merged model
print(f"Saving merged model to {merged_model_output_path}")
merged_model.save_pretrained(merged_model_output_path)
tokenizer.save_pretrained(merged_model_output_path)

# Cleanup
del base_model, lora_model, merged_model
gc.collect()
torch.cuda.empty_cache()

print("✅ Merge complete.")
