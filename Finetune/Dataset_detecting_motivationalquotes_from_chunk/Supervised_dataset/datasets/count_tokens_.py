from transformers import AutoTokenizer
import json

# Path to your local tokenizer/model directory
model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct-MERGED"

tokenizer = AutoTokenizer.from_pretrained(model_path)

def count_tokens(text: str) -> int:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if not lines:
    print("File is empty!")
else:
    total_tokens = 0
    max_tokens = 0
    max_line_num = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON on line {i}: {e}")
            continue

        messages = obj.get("messages", [])
        combined_content = " ".join(msg.get("content", "") for msg in messages)

        tokens_count = count_tokens(combined_content)

        total_tokens += tokens_count
        if tokens_count > max_tokens:
            max_tokens = tokens_count
            max_line_num = i

        print(f"Line {i}: combined content tokens = {tokens_count}")

    print(f"\nTotal lines processed: {len(lines)}")
    print(f"Max tokens in a single line: {max_tokens} (line {max_line_num})")
    print(f"Average tokens per line: {total_tokens / len(lines):.2f}")
