from transformers import AutoTokenizer
import json
import os
import logging
from datetime import datetime

# Paths
model_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-Instruct-finetuned-Motivational-text"
dataset_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\new_verify_agent_dataset.jsonl"
tokenizer_config_path = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-Instruct-finetuned-Motivational-text\tokenizer_config.json"
log_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\logs\token_count_log.txt"

# Set up logging
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load chat template from tokenizer_config.json
if not os.path.exists(tokenizer_config_path):
    logger.error(f"Tokenizer config file not found at {tokenizer_config_path}")
    raise FileNotFoundError(f"Tokenizer config file not found at {tokenizer_config_path}")
with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
chat_template = tokenizer_config.get("chat_template")
if not chat_template:
    logger.error("Chat template not found in tokenizer_config.json")
    raise ValueError("Chat template not found in tokenizer_config.json")
logger.info("Loaded chat template from tokenizer_config.json")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.chat_template = chat_template
    tokenizer.eos_token = "<|end|>"  # Match fine_tune_mistrail_copy.py
    tokenizer.pad_token = "<|endoftext|>" if tokenizer.pad_token is None else tokenizer.pad_token
    logger.info(f"Tokenizer loaded: eos_token={tokenizer.eos_token}, pad_token={tokenizer.pad_token}")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

def count_tokens(messages: list[dict]) -> int:
    """Count tokens for a list of messages using the chat template."""
    try:
        # Apply chat template to format messages (without generation prompt for training data)
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # Tokenize with special tokens
        tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        logger.error(f"Error tokenizing messages: {e}")
        return 0

# Read dataset
logger.info(f"Processing dataset: {dataset_path}")
if not os.path.exists(dataset_path):
    logger.error(f"Dataset file not found at {dataset_path}")
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
with open(dataset_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if not lines:
    logger.warning("Dataset file is empty!")
    print("File is empty!")
else:
    total_tokens = 0
    max_tokens = 0
    max_line_num = 0
    valid_lines = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            logger.info(f"Skipping empty line {i}")
            print(f"Skipping empty line {i}")
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON on line {i}: {e}")
            print(f"Invalid JSON on line {i}: {e}")
            continue

        messages = obj.get("messages", [])
        if not messages or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
            logger.error(f"Invalid message structure on line {i}")
            print(f"Invalid message structure on line {i}")
            continue

        tokens_count = count_tokens(messages)
        if tokens_count == 0:
            logger.warning(f"No tokens counted for line {i}")
            print(f"No tokens counted for line {i}")
            continue

        valid_lines += 1
        total_tokens += tokens_count
        if tokens_count > max_tokens:
            max_tokens = tokens_count
            max_line_num = i

        logger.info(f"Line {i}: token count = {tokens_count}")
        print(f"Line {i}: token count = {tokens_count}")

    # Summary statistics
    logger.info(f"Total lines processed: {len(lines)}")
    logger.info(f"Valid lines processed: {valid_lines}")
    logger.info(f"Max tokens in a single example: {max_tokens} (line {max_line_num})")
    logger.info(f"Average tokens per valid example: {total_tokens / valid_lines:.2f}" if valid_lines > 0 else "No valid examples")
    recommended_max_seq_length = max_tokens + int(max_tokens * 0.1)
    logger.info(f"Recommended max_seq_length: {recommended_max_seq_length} (highest token count + 10% buffer)")
    
    print(f"\nTotal lines processed: {len(lines)}")
    print(f"Valid lines processed: {valid_lines}")
    print(f"Max tokens in a single example: {max_tokens} (line {max_line_num})")
    print(f"Average tokens per valid example: {total_tokens / valid_lines:.2f}" if valid_lines > 0 else "No valid examples")
    print(f"Recommended max_seq_length: {recommended_max_seq_length} (highest token count + 10% buffer)")