import os
import torch
from log import merge_logger
from peft import PeftModel,AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter_checkpoint(base_model_path, adapter_path, output_dir):
    """
    Merges the LoRA adapter weights into the base model and saves the resulting model.

    Args:
        base_model_path (str): Path to the base model directory.
        adapter_path (str): Path to the LoRA adapter checkpoint directory.
        output_dir (str): Directory where the merged model will be saved.
    """

    merge_logger("🔄 Loading tokenizer from adapter checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)  # KEY CHANGE: Use adapter's tokenizer
    
    merge_logger("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16
    )
    merge_logger(f"Base model vocab size: {base_model.config.vocab_size}")

    merge_logger("🔄 Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merge_logger(f"✅ LoRA adapter loaded from {adapter_path}")

    merge_logger("🔄 Merging LoRA adapter weights into base model...")
    merged_model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merge_logger("✅ LoRA adapter weights merged successfully")

    merge_logger(f"🔄 Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)  # Save the correct tokenizer
    merge_logger(f"✅ Merged model and tokenizer saved to: {output_dir}")

#if __name__ == "__main__":
    # base_model = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen2.5-Coder-3B-Instruct"
    # adapter = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen2.5-Coder-3B-Instruct\checkpoint-5"
    # merged_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen2.5-Coder-3B-Instruct\merged_model_finetuned"
    
    # merge_adapter_checkpoint(
    #     base_model_path=base_model,
    #     adapter_path=adapter,
    #     output_dir=merged_path
    # )


def merge_adapter_checkpoint_2(adapter_path: str, save_path: str, base_model: str = None):
    """
    Laster base-modell + adapter, merger adapter-vekt inn i modellen, og lagrer alt som én modell.
    adapter_path: sti til adapter-checkpoint (inneholder peft_config og weights)
    save_path: hvor den merge'ede modellen skal lagres
    base_model: hvis du ønsker eksplisitt angivelse av base-modell
    """
    # ⭐ Metode A: praktisk direkte load
    peft_model = AutoPeftModelForCausalLM.from_pretrained(adapter_path)
    merge_logger(type(peft_model))
    # ⭐ Metode B: manuell base + adapter
    # base = AutoModelForCausalLM.from_pretrained(base_model)
    # peft_model = PeftModel.from_pretrained(base, adapter_path)

    # Merger adapter-vekt inn i selve modellen
    merged_model = peft_model.merge_and_unload()
    merge_logger(type(merged_model))

    # Lagrer merged model og tokenizer
    merged_model.save_pretrained(save_path)
    tok = AutoTokenizer.from_pretrained(adapter_path)
    tok.save_pretrained(save_path)

    merge_logger(f"✅ Flettet modell lagret i: {save_path}")

# if __name__ == "__main__":
#     merge_adapter_checkpoint(
#         adapter_path="./your_adapter_checkpoint",
#         save_path="./merged_full_model"
#     )
