# import json
# from transformers import AutoTokenizer
# from tqdm import tqdm





# def trunacte_dataset(input_jsnol):
#     model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct"
#     output_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\traintest.jsonl"
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

#     max_length = 1700
#     kept_count = 0
#     removed_count = 0

#     with open(input_jsnol, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#         for line in tqdm(infile, desc="Filtering examples"):
#             data = json.loads(line)
#             if "messages" not in data:
#                 continue
#             try:
#                 # Create prompt using chat template (no immediate tokenization)
#                 prompt = tokenizer.apply_chat_template(data["messages"], tokenize=False)
#                 tokens = tokenizer(prompt, return_tensors="pt").input_ids
#                 length = tokens.shape[1]
#                 if length <= max_length:
#                     outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#                     kept_count += 1
#                 else:
#                     removed_count += 1
#             except Exception as e:
#                 print(f"Error processing example: {e}")
#                 continue

#     print(f"Filtering complete. Kept: {kept_count} examples. Removed: {removed_count} examples.")

# if __name__ == "__main__":

#     trunacte_dataset(input_jsnol=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl")import json
from transformers import AutoTokenizer
from tqdm import tqdm
import json

def filter_dataset(input_jsonl):
    model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-instruct"
    output_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\traintest.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    min_length = 0
    max_length = 2500
    kept_count = 0
    removed_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Filtering examples"):
            try:
                data = json.loads(line)
                if "messages" not in data:
                    removed_count += 1
                    continue

                # Apply chat template without tokenizing first
                prompt = tokenizer.apply_chat_template(data["messages"], tokenize=False)
                tokens = tokenizer(prompt, return_tensors="pt").input_ids
                length = tokens.shape[1]

                if min_length < length <= max_length:
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    kept_count += 1
                else:
                    removed_count += 1
            except Exception as e:
                print(f"Error processing example: {e}")
                removed_count += 1

    print(f"Filtering complete. Kept: {kept_count} examples. Removed: {removed_count} examples.")

if __name__ == "__main__":
    filter_dataset(
        input_jsonl=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\test.jsonl"
    )
