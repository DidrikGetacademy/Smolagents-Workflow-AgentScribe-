from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def get_param_diffs(model_before, model_after, tol=1e-6):
    diffs = []
    for (name_before, param_before), (name_after, param_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
        if name_before != name_after:
            print(f"Warning: parameter names mismatch: {name_before} vs {name_after}")
            continue
        if not torch.allclose(param_before.cpu(), param_after.cpu(), atol=tol):
            diffs.append(name_before)
    return diffs

base_model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-7B-Instruct-v0.3"
lora_adapter_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-7B-Instruct-v0.3\checkpoint-50"
merged_model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-7B-Instruct-v0.3\Merged_model"

print("Loading tokenizer from finetuned folder (with added tokens)...")
tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, local_files_only=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, device_map="cuda", torch_dtype=torch.bfloat16, local_files_only=True
)

print("Resizing token embeddings to match tokenizer vocab size...")
base_model.resize_token_embeddings(len(tokenizer))

# Clone base model weights before merging for comparison
base_model_before_weights = {k: v.clone().cpu() for k, v in base_model.named_parameters()}

print("Loading LoRA adapter...")
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging LoRA weights into base model...")
lora_model = lora_model.merge_and_unload()

# Compare weights and find changed layers
diff_layers = []
for name, param in lora_model.named_parameters():
    if name in base_model_before_weights:
        if not torch.allclose(param.cpu(), base_model_before_weights[name], atol=1e-6):
            diff_layers.append(name)

print(f"Layers changed by LoRA merging ({len(diff_layers)} layers):")
for layer in diff_layers:
    print(layer)

print("Saving merged model...")
lora_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved to {merged_model_path}")
