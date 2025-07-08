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

def add_timestamps_and_filler(sentences, filler_before, filler_after, base_start=600.0, avg_sentence_duration=3.0):
    all_sents = filler_before + sentences + filler_after
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
    sents = re.split(r'(?<=[.!?]) +', quote)
    return [s.strip() for s in sents if s.strip()]
def save_as_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_filler_sentences():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\filler_sentence_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        filler_sentences = [entry["sentence"].strip() for entry in data if 20 < len(entry["sentence"].strip()) < 150]
    log(f"[Dataset Load] Loaded {len(filler_sentences)} filler sentences from local JSON")
    return filler_sentences


def sample_unique_fillers(n, filler_pool, used_set):
    available = list(set(filler_pool) - used_set)
    if len(available) == 0:
        return None
    n = min(n, len(available))
    selected = random.sample(available, n)
    used_set.update(selected)
    return selected

def chunk_text_with_filler_quote_split(filler_sentences, quote, used_filler_sentences, base_start=600.0, avg_sentence_duration=3.0):
    # Del opp quote i ord
    quote_words = quote.split()
    # Ta 1-3 ord til å ha med på filler-linja
    n_quote_start = random.randint(1, min(3, len(quote_words)))
    quote_start_words = quote_words[:n_quote_start]
    quote_rest_words = quote_words[n_quote_start:]

    # Velg filler setning (1 setning)
    filler = sample_unique_fillers(1, filler_sentences, used_filler_sentences)
    if filler is None:
        return None
    filler = filler[0]

    lines = []
    current_time = base_start

    # Første linje: filler + 1-3 ord quote på samme linje
    line1_text = filler.strip() + " " + " ".join(quote_start_words)
    duration = random.uniform(avg_sentence_duration * 0.8, avg_sentence_duration * 1.2)
    start_t = current_time
    end_t = current_time + duration
    lines.append(f"[{start_t:.2f}s - {end_t:.2f}s] {line1_text}")
    current_time = end_t + 0.1

    # Resten av quote på egne linjer (split i setninger evt)
    rest_text = " ".join(quote_rest_words)
    # Del opp i setninger (for eksempel split på punktum)
    rest_sents = re.split(r'(?<=[.!?]) +', rest_text) if rest_text else []

    for sent in rest_sents:
        sent = sent.strip()
        if not sent:
            continue
        duration = random.uniform(avg_sentence_duration * 0.8, avg_sentence_duration * 1.2)
        start_t = current_time
        end_t = current_time + duration
        lines.append(f"[{start_t:.2f}s - {end_t:.2f}s] {sent}")
        current_time = end_t + 0.1

    # Returnerer hele chunk_text som string med linjeskift
    chunk_text = "[chunk start]\n" + "\n".join(lines) + "\n[chunk end]\n"
    return {
    "chunk_text": chunk_text,
    "quote_lines": lines[1:],  # Dette er de "rene" quote-linjene
    "quote_start_line": lines[0],  # Første linje med filler + start
    "n_quote_start": n_quote_start,  # Antall quote-ord i første linje
}

def preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, dataset_quotes=None):
   
    instruction = """
        You are a quote‐detection assistant for creating motivational shorts. Your task is to carefully analyze and read a timestamped chunk and extract any standalone motivational quotes or advice or motivational passages that are complete and self‐contained, and could be used for a short inspirational video, If quotes are found, save them using the appropriate function. If none are found, return a final response indicating that.
        You are to Save standalone motivational quotes, advice, inspiring messages, that are  complete  and does not lack context if isolated from the rest of the chunk.
        Analyze the chunk/text between [chunk start] and [chunk end].

        ### Objective:
        Your job is to extract motivational quotes from the input text chunk. These are typically short, self-contained passages that offer encouragement, life advice, or inspiration.

        ###  Reasoning:
        Always begin with a `Thought:` statement explaining your reasoning — for example, whether you identified quotes, and how many.

        ###Instructions --- Your Expected Output Format:
        - If **two quote, Advice or motivational complete message** is found in the chunk you analyze, output:
            Thought: I found 2 standalone motivational passages that meet the criteria, so I’m saving them.
            <code>
            SaveMotivationalText(text="[start - end] Quote 1", text_file=text_file)
            SaveMotivationalText(text="[start - end] Quote 2", text_file=text_file)
            final_answer("im done analyzing chunk")
            </code>

        - If **one  quote, Advice or motivational complete message** is found in the chunk you analyze, output:
            Thought: I found 1 standalone motivational passage that meets the criteria, so I’m saving it.
            <code>
            SaveMotivationalText(text="[start - end] Quote", text_file=text_file)
            final_answer("im done analyzing chunk")
            </code>
            
                
        - If **no quotes, Advice or motivational complete message** is found in the chunk you analyze, output:
            Thought:Thought: I carefully scanned every timestamped line in this chunk, looking for a short, self‑contained motivational passage. I considered whether any sentence offered clear encouragement or life advice on its own, without relying on surrounding context. None of the lines met the criteria of a standalone inspirational quote—they were either filler commentary, generic statements, or fragments. Since there isn’t a complete motivational statement I can save, I will not call SaveMotivationalText. and only provide `final_answer`
            <code>
            final_answer("After carefully analyzing the chunk/text, I have concluded nothing can be saved.")
            </code>

        ### Notes:
        - Quotes must be **motivational** and **standalone** — avoid fragments or generic sentences.
        - Always include both `Thought:` and `<code>` blocks.
        - Use exact function names and punctuation as shown.
        - Do not return quotes that are incomplete or unclear.
        - o not create multiple SaveMotivationalText() calls for each line in a single quote.
        - Do not alter or guess missing timestamps — use the exact start and end values provided in the lines that contain the quote.
        - Quote text should appear as a single, continuous string, even if it was originally split across 2–3 lines.
                        
        ##Timestamp Handling:
        When a quote spans multiple lines (each line containing a separate timestamp):
            - Merge the lines into a single quote.
            - Include the **start time from the first line** and the **end time from the last line**.
            - Preserve original spacing and punctuation.
            - Output the full quote like:

        Exsample identified quote from chunk:
        [chunk start]
        [620.10s - 622.40s] In today's episode, we'll cover some important updates about mental clarity
        [622.41s - 623.69s] But before that, thank you for supporting the channel
        [623.70s - 627.11s] You will encounter many challenges in life  
        [627.12s - 628.00s] But you must never be defeated by the challenges
        [628.01s - 629.55s] That was a quote I heard recently and it really stuck with me
        [629.56s - 631.00s] Anyway, let’s move on to the main topic of today’s discussion
        [chunk end]

        Your output should be:

        Thought: I found 1 standalone motivational passages that meet the criteria, so I’m saving them.
        <code>SaveMotivationalText(text="[623.70s - 628.00s] You will encounter many challenges in life  [627.12s - 628.00s] But you must never be defeated by the challenges", text_file=text_file) final_answer("Im done analyzing the chunk")</code>\n

        Here is the chunk you will analyze:\n


    """
    if idx % 3 == 0:
        n_fillers = random.randint(8, 15)
        selected_fillers = sample_unique_fillers(n_fillers, filler_sentences, used_filler_sentences)
        if selected_fillers is None:
            return None

        chunk_text = "[chunk start]\n" + add_timestamps_and_filler([], selected_fillers, []) + "[chunk end]\n"
        label_text = (
            'Thought: I carefully scanned every timestamped line in this chunk, looking for a short, self‑contained motivational passage. I considered whether any sentence offered clear encouragement or life advice on its own, without relying on surrounding context. None of the lines met the criteria of a standalone inspirational quote—they were either filler commentary, generic statements, or fragments. Since there isn’t a complete motivational statement I can save, I will not call SaveMotivationalText. and only provide `final_answer` '
            '<code>\n'
            'final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved.")\n'
            '</code>'
        )
    elif idx % 2 == 0:
        # Velg quote og filler
        quote = example["quote"]
        chunk_info = chunk_text_with_filler_quote_split(filler_sentences, quote, used_filler_sentences)

        chunk_text = chunk_info["chunk_text"]
        if chunk_text is None:
            return None
        quote_lines = chunk_info["quote_lines"]
        quote_start_line = chunk_info["quote_start_line"]
        n_quote_start = chunk_info["n_quote_start"]




        match = re.match(r"\[[^\]]+\]\s+(.*)", quote_start_line)
        if match:
            text = match.group(1)
            words = text.split()
            quote_start_words = words[-n_quote_start:]  
            timestamp = quote_start_line.split("]")[0] + "]"
            quote_start_cleaned = f"{timestamp} {' '.join(quote_start_words)}"
        else:
            quote_start_cleaned = ""

      
        full_quote_lines = [quote_start_cleaned] + quote_lines
        full_quote = " ".join(full_quote_lines).replace('"', '\\"')

        label_text = (
            'Thought: After detecting and understanding connections between lines i identified a motivational quote/advice. I found 1 standalone motivational passage that meet the criteria, so I’m saving it.\n' +
            '<code>\n'
            f'SaveMotivationalText(text="{full_quote}", text_file=text_file)\n'
            'final_answer("Im done analyzing the chunk")\n'
            '</code>'
        )
        return {
            "messages": [
                {"role": "user", "content": instruction.strip() + "\n\n" + chunk_text.strip()},
                {"role": "assistant", "content": label_text.strip()}
            ]
        }

    else:
        double_quote_chance = 0.5
        if dataset_quotes is None:
            dataset_quotes = [example["text"]]

        def clean_quote(q):
            return q.replace("“", '').replace("”", '').replace("‘", "").replace("’", "").replace(".", "").replace('""', '').strip()

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
                n_fillers_mid = random.randint(1, 6)
                fillers_mid = sample_unique_fillers(n_fillers_mid, filler_sentences, used_filler_sentences)
                if fillers_mid is None:
                    return None
                all_sentences.extend(fillers_mid)

        filler_before = sample_unique_fillers(random.randint(1, 10), filler_sentences, used_filler_sentences)
        filler_after = sample_unique_fillers(random.randint(1, 7), filler_sentences, used_filler_sentences)
        if filler_before is None or filler_after is None:
            return None

        chunk_text = add_timestamps_and_filler(all_sentences, filler_before, filler_after)
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
                'Thought: I found 1 standalone motivational passage that meet the criteria, so I’m saving it.\n' +
                '<code>\n'
                f'SaveMotivationalText(text="{full_quote_with_timestamps}", text_file=text_file)\n'
                'final_answer("Im done analyzing the chunk")\n'
                '</code>'
            )
        else:
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
                save_calls.append(f'SaveMotivationalText(text="{combined_text}", text_file=text_file)')
            Thought = f"I found {len(save_calls)} standalone motivational passage{'s' if len(save_calls)>1 else ''} that meet the criteria, so I’m saving {'them' if len(save_calls)>1 else 'it'}."
            label_text = (
                f"Thought: {Thought}\n"
                "<code>\n" + "\n".join(save_calls) + "\n"
                'final_answer("Im done analyzing the chunk")\n</code>'
            )

    return {
        "messages": [
            {"role": "user", "content": instruction.strip() + "\n\n" + chunk_text.strip()},
            {"role": "assistant", "content": label_text.strip()}
        ]
    }
def load_quote_dataset():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\Quote_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quotes = [entry["quote"].strip() for entry in data if entry["quote"].strip()]
    return quotes

def main():
    used_filler_sentences = set()

    filler_sentences = load_filler_sentences()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    from sklearn.model_selection import train_test_split

    quotes_list = load_quote_dataset()
    dataset = [{"quote": q} for q in quotes_list]

    train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=42)


    quotes_list = [x['quote'] for x in dataset]

    train_data = []
    test_data = []

    for idx, example in enumerate(train_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, quotes_list)
        if processed is None:
            break
        train_data.append(processed)

    for idx, example in enumerate(test_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, quotes_list)
        if processed is None:
            break
        test_data.append(processed)

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("test.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log("Saved train.json and test.json with processed data.")

if __name__ == "__main__":
    main()
