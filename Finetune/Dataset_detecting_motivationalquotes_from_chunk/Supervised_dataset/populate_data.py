import json
import random
import re
from datasets import load_dataset
from transformers import AutoTokenizer

CONFIG = {
    'model_name': 'mistralai/Mistral-7B-Instruct-v0.3',
}

def log(msg):
    print(msg)

def add_timestamps_and_filler(sentences, filler_sentences, base_start=600.0, avg_sentence_duration=3.0):
    all_sents = filler_sentences + sentences + filler_sentences
    timestamped_text = ""
    current_time = base_start
    for sent in all_sents:
        duration = random.uniform(avg_sentence_duration * 0.8, avg_sentence_duration * 1.2)
        start_time = current_time
        end_time = current_time + duration
        timestamped_text += f"[{start_time:.2f}s - {end_time:.2f}s] {sent.strip()}\n"
        current_time = end_time + 0.1
    return timestamped_text

def split_quote_randomly(quote):
    # Simple split on sentences, could be improved
    sents = re.split(r'(?<=[.!?]) +', quote)
    return [s.strip() for s in sents if s.strip()]

def load_filler_sentences():
    dataset = load_dataset("ylacombe/podcast_fillers_processed",  split="train")
    dataset = dataset.shuffle(seed=42)
    filler_sentences = []
    for text in dataset['text']:
        sents = text.split('.')
        for s in sents:
            s = s.strip()
            if 20 < len(s) < 150:
                filler_sentences.append(s + '.')
    log(f"[Dataset Load] Loaded {len(filler_sentences)} filler sentences")
    return filler_sentences

def preprocess_with_filler_balanced(example, idx, filler_sentences, tokenizer, dataset_quotes=None):


    instruction = """
    Save standalone motivational quotes.
    Analyze the chunk between [chunk start] and [chunk end].
    If you find one or more qualifying quotes, provide a Thought explaining your analysis, save each quote separately with SaveMotivationalText including all timestamps for that quote, and finish with final_answer().
    If no quotes are found, provide a Thought stating no suitable quotes were found, and just use final_answer().
    """

    system = """
    You are an expert assistant analyzing text using chain-of-thought reasoning and Python tool calls.
    You will be given a task and access to tools you call with code blocks.
    Your process follows cycles of 'Thought:' explaining your reasoning, and '<code>' blocks with code that end with '</code>'.
    At the end, always return a final answer with the `final_answer` tool.

    Your task:
    - Carefully analyze the chunk sentence by sentence.
    - Extract standalone motivational quotes or passages that form complete, inspirational thoughts.
    - Preserve and include all exact timestamps for every sentence in each extracted quote.
    - For each extracted quote, output a separate SaveMotivationalText call with the full text including timestamps.
    - After processing all quotes, output a final_answer indicating completion.
    - If no motivational quotes are found, output a Thought explaining this and a final_answer stating that nothing was saved.

    Output format examples:

    If there is only one motivational quote:
    Thought: I found 1 motivational passage that meets the criteria, so I’m saving it.
    <code>
    SaveMotivationalText(text="[start-end] motivational quote sentence 1 [start-end] sentence 2 ...", text_file=text_file)
    final_answer("I have analyzed the chunk and saved 1 motivational quote.")
    </code>

    If there are multiple motivational quotes:
    Thought: I found N motivational passages that meet the criteria, so I’m saving them.
    <code>
    SaveMotivationalText(text="[start-end] motivational quote 1 sentence 1 ...", text_file=text_file)
    SaveMotivationalText(text="[start-end] motivational quote 2 sentence 1 ...", text_file=text_file)
    final_answer("I have analyzed the chunk and saved N motivational quotes.")
    </code>

    If none found:
    Thought: No standalone motivational quotes found in this chunk.
    <code>
    final_answer("After careful analysis, no quotes saved.")
    </code>
    """
    # Every 3rd chunk: filler only
    if idx % 3 == 0:
        n_fillers = random.randint(5, 10)
        selected_fillers = random.sample(filler_sentences, n_fillers)
        chunk_text = "[chunk start]\n" + add_timestamps_and_filler([], selected_fillers) + "[chunk end]\n"
        label_text = (
            'Thought: No Standalone Quote found in the chunk. so i will not save anything and just provide a final answer\n'
            '<code>\n'
            'final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved.")\n'
            '</code>'
        )
    else:
        double_quote_chance = 0.5
        if dataset_quotes is None:
            dataset_quotes = [example["quote"]]

        def clean_quote(q):
            return q.replace("“", '').replace("”", '').replace("‘", "").replace("’", "").replace(".","").replace('""', '').strip()

        if random.random() < double_quote_chance and len(dataset_quotes) > 1:
            other_example = random.choice(dataset_quotes)
            while other_example == example["quote"]:
                other_example = random.choice(dataset_quotes)
            quotes = [clean_quote(example["quote"]), clean_quote(other_example)]
        else:
            quotes = [clean_quote(example["quote"])]

        all_sentences = []
        quotes_splits = []
        for i, q in enumerate(quotes):
            quote_sents = split_quote_randomly(q)
            quotes_splits.append(quote_sents)
            all_sentences.extend(quote_sents)
            if i < len(quotes) - 1:
                n_fillers_mid = random.randint(1, 3)
                fillers_mid = random.sample(filler_sentences, n_fillers_mid)
                all_sentences.extend(fillers_mid)

        n_fillers_before = random.randint(1, 5)
        n_fillers_after = random.randint(1, 5)
        filler_before = random.sample(filler_sentences, n_fillers_before)
        filler_after = random.sample(filler_sentences, n_fillers_after)

        chunk_text = add_timestamps_and_filler(all_sentences, filler_before + filler_after)
        chunk_text = "[chunk start]\n" + chunk_text + "[chunk end]\n"
        chunk_lines = [line.strip() for line in chunk_text.strip().splitlines() if line.strip()]
        timestamp_re = re.compile(r"\[(\d+\.\d+)s - (\d+\.\d+)s\] (.+)")
        parsed_lines = []
        for line in chunk_lines:
            m = timestamp_re.match(line)
            if m:
                start_t = float(m.group(1))
                end_t = float(m.group(2))
                text = m.group(3).strip()
                parsed_lines.append((start_t, end_t, text))

        flat_quote_lines = []
        for quote_chunk in quotes_splits:
            flat_quote_lines.extend(quote_chunk)

        matched_quote_lines = []
        pi = 0
        for q_line in flat_quote_lines:
            found = False
            q_line_clean = q_line.strip().lower()
            while pi < len(parsed_lines):
                start_t, end_t, line_text = parsed_lines[pi]
                line_text_clean = line_text.strip().lower()
                if q_line_clean in line_text_clean or line_text_clean in q_line_clean:
                    matched_quote_lines.append((start_t, end_t, line_text))
                    pi += 1
                    found = True
                    break
                pi += 1
            if not found:
                matched_quote_lines.append((None, None, q_line))

        if len(quotes) == 1:
            parts = []
            for start_t, end_t, text in matched_quote_lines:
                if start_t is not None and end_t is not None:
                    parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                else:
                    parts.append(text)
            full_quote_with_timestamps = " ".join(parts).replace('"', '\\"')
            label_text = (
                'Thought: I found 1 standalone motivational passagethat meet the criteria, so I’m saving it.\n' +
                '<code>\n'
                f'SaveMotivationalText(text="{full_quote_with_timestamps}",text_file=text_file)\n'
                'final_answer("im done analyzing chunk")\n'
                '</code>'
            )
        else:
            label_text = ""
            save_calls = []
            idx_line = 0
            for quote_chunk in quotes_splits:
                parts = []
                for _ in quote_chunk:
                    start_t, end_t, text = matched_quote_lines[idx_line]
                    idx_line += 1
                    if start_t is not None and end_t is not None:
                        parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                    else:
                        parts.append(text)
                combined_text = " ".join(parts).replace('"', '\\"')
                save_calls.append(f'SaveMotivationalText(text="{combined_text}",text_file=text_file)')
            Thought = f"I found {len(save_calls)} standalone motivational passage{'s' if len(save_calls)>1 else ''} that meet the criteria, so I’m saving {'them' if len(save_calls)>1 else 'it'}."
            label_text = (
                f"Thought: {Thought}\n"
                "<code>\n"
                + "\n".join(save_calls) + "\n"
                'final_answer("im done analyzing chunk")\n'
                "</code>"
            )

    return {
        "system_prompt": system,
        "instruction": instruction,
        "input_text": chunk_text,
        "label_text": label_text
    }

def main():
    filler_sentences = load_filler_sentences()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)

    dataset = load_dataset("asuender/motivational-quotes", 'quotes', split='train')
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_datasets['train']
    test_ds = split_datasets['test']

    quotes_list = [x['quote'] for x in dataset]

    train_data = []
    test_data = []

    # Process train examples
    for idx, example in enumerate(train_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, tokenizer, quotes_list)
        train_data.append(processed)

    # Process test examples
    for idx, example in enumerate(test_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, tokenizer, quotes_list)
        test_data.append(processed)

    # Save to json files
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\train.jsonl", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\test.jsonl", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    log("Saved train.json and test.json with processed data.")

if __name__ == "__main__":
    main()
