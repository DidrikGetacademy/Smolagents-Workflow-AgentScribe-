import json
import random
import re
from datasets import load_dataset
from transformers import AutoTokenizer
import re

CONFIG = {
    'model_name': r'C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-3B-Instruct',
}
count_only_filler = 0
count_single_Quote_mix = 0
count_single_quote_natural = 0
Count_double_quotes = 0
def log(msg):
    print(msg)

def add_timestamps_and_filler(sentences, filler_before, filler_after, base_start=random.uniform(300.0, 1200.0), avg_sentence_duration=3.0):
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

def split_quote_randomly(quote, max_words_per_line=10):
    words = quote.strip().split()
    chunks = []
    i = 0
    while i < len(words):
        n = random.randint(4, max_words_per_line)
        chunk = words[i:i+n]
        if chunk:
            chunks.append(" ".join(chunk))
        i += n
    return chunks

def save_as_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_filler_sentences():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\filler_sentence_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        filler_sentences = [entry["sentence"].strip() for entry in data if 20 < len(entry["sentence"].strip()) < 150]
    log(f"[Dataset Load] Loaded {len(filler_sentences)} filler sentences from local JSON")
    return filler_sentences


def extract_quote_only_lines(quote_start_line, quote_lines, n_quote_start):
    # Del opp første linje i ord
    match = re.match(r"\[(.*?)s - (.*?)s\] (.+)", quote_start_line)
    if not match:
        return []
    start_t, end_t, line_text = match.groups()
    words = line_text.strip().split()
    quote_words = words[-n_quote_start:]  # De siste ordene er quote
    quote_text = " ".join(quote_words)
    quote_start_clean = f"[{float(start_t):.2f}s - {float(end_t):.2f}s] {quote_text}"

    return [quote_start_clean] + quote_lines

def sample_unique_fillers(n, filler_pool, used_set):
    available = list(set(filler_pool) - used_set)
    if not available:
        return []
    selected = random.sample(available, min(n, len(available)))
    used_set.update(selected)
    return selected




def preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, dataset_quotes=None):
    global count_only_filler
    global count_single_Quote_mix
    global count_single_quote_natural
    global Count_double_quotes
    system_prompt = """
        You are a quote‐detection assistant for creating motivational shorts. Your task is to carefully analyze and read a timestamped chunk and extract any standalone motivational quotes or advice or motivational passages that are complete and self‐contained, and could be used for a short inspirational video, If quotes are found, save them using the appropriate function. If none are found, return a final response indicating that.
        You are to Save standalone motivational quotes, advice, inspiring messages, that are  complete  and does not lack context if isolated from the rest of the chunk.
        Analyze the chunk/text between [chunk start] and [chunk end].
        Objective: Your job is to extract motivational quotes from the input text chunk. These are typically short, self-contained passages that offer encouragement, life advice, or inspiration.
        Reasoning: Always begin with a `Thought:` statement explaining your reasoning — for example, whether you identified quotes, and how many.
        Instructions --- Your Expected Output Format:
        - If two quote, Advice or motivational complete message is found in the chunk you analyze, output:
            Thought: I found 2 standalone motivational passages that meet the criteria, so I’m saving them.
            <code>
            SaveMotivationalText(text="[start - end] Quote 1", text_file=text_file)
            SaveMotivationalText(text="[start - end] Quote 2", text_file=text_file)
            final_answer("im done analyzing chunk")
            </code>
        - If one  quote, Advice or motivational complete message is found in the chunk you analyze, output:
            Thought: I found 1 standalone motivational passage that meets the criteria, so I’m saving it.
            <code>
            SaveMotivationalText(text="[start - end] Quote", text_file=text_file)
            final_answer("im done analyzing chunk")
            </code>    
        - If no quotes, Advice or motivational complete message is found in the chunk you analyze, output:
            Thought:Thought: I carefully scanned every timestamped line in this chunk, looking for a short, self‑contained motivational passage. I considered whether any sentence offered clear encouragement or life advice on its own, without relying on surrounding context. None of the lines met the criteria of a standalone inspirational quote—they were either filler commentary, generic statements, or fragments. Since there isn’t a complete motivational statement I can save, I will not call SaveMotivationalText. and only provide `final_answer`
            <code>
            final_answer("After carefully analyzing the chunk/text, I have concluded nothing can be saved.")
            </code>
        Notes:
        - Quotes must be motivational and standalone — avoid fragments or generic sentences.
        - Always include both `Thought:` and `<code>` blocks.
        - Use exact function names and punctuation as shown.
        - Do not return quotes that are incomplete or unclear.
        - Do not create multiple SaveMotivationalText() calls for each line in a single quote.
        - Do not alter or guess missing timestamps — use the exact start and end values provided in the lines that contain the quote.
        - Quote text should appear as a single, continuous string, even if it was originally split across 2–3 lines.         
        Timestamp Handling:
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
    """
   
    instruction = """
        Here is the chunk you will analyze:\n
    """

    if random.random() < 0.2:
        count_only_filler += 1
        n_fillers = random.randint(5, 10)
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

    else:
        quote_filler_mix_bool = False
        quote_filler_mix_chance = 0.5
        double_quote_chance = 0.5
  
        if dataset_quotes is None:
            dataset_quotes = [example["text"]]

        def clean_quote(q):
            return q.replace("“", '').replace("”", '').replace("‘", "").replace("’", "").replace(".", "").replace('""', '').strip()

        if random.random() < double_quote_chance and len(dataset_quotes) > 1:
            Count_double_quotes += 1
            other_example = random.choice(dataset_quotes)
            while other_example == example["quote"]:
                other_example = random.choice(dataset_quotes)
            quotes = [clean_quote(example["quote"]), clean_quote(other_example)]

        else:
            quotes = [clean_quote(example["quote"])]

        all_sentences = []
        quotes_splits = []
        quote_filler_mix_bool = (len(quotes) == 1) and (random.random() < quote_filler_mix_chance)
        for i, q in enumerate(quotes):
            random_max_words = random.randint(4,7)

            # Split quote i mindre linjer som vanlig
            quote_sents = split_quote_randomly(q, max_words_per_line=random_max_words)
            print(f"Spltted quotes randomaly: {quote_sents}")

            if quote_filler_mix_bool and len(quotes) == 1:
                count_single_Quote_mix += 1
                #henter ut en filler linje
                filler_intro = sample_unique_fillers(1, filler_sentences, used_filler_sentences)
                if filler_intro is not None and len(filler_intro) > 0: # hvis filler_intro ikke er null og mengden filler_intro er større enn 0
                  #  start_end_mix_chance = 0.5
                   # if random.random() > start_end_mix_chance:
                        
                    quote_words = quote_sents[0].split() # splitter en quote linje fra første linja/starten av quoten i lista til ord.
                    quote_word_amount = random.randint(1, len(quote_words))
                        
                    moved_words = " ".join(quote_words[:quote_word_amount])
                    remaining_quote = " ".join(quote_words[quote_word_amount:])

                    #slå sammen filler og første quote-linje
                    randomkChance = 0.5
                    if random.random() < randomkChance:

                        mixed_line = filler_intro[0].rstrip(".") + ", " + moved_words
                    else: 
                        mixed_line = filler_intro[0].rstrip(".") + " " + moved_words

                    if remaining_quote:
                        quote_sents[0] = mixed_line
                        quote_sents.insert(1, remaining_quote)
                    else:
                        quote_sents[0] = mixed_line
                    # else:
                    #     quote_words = quote_sents[-1].split() 
                    #     quote_word_amount = random.randint(1,len(quote_words) - 2)

                    #     moved_words = " ".join(quote_words[-quote_word_amount:])
                    #     remaining_quote = " ".join(quote_words[:-quote_word_amount])

                    #     randomkChance = 0.5
                    #     if random.random() < randomkChance:
                    #         mixed_filler = moved_words + ", " + filler_intro[0].lstrip().capitalize()
                    #     else:
                    #         mixed_filler = moved_words + " " + filler_intro[0].lstrip().capitalize()

                    #     if remaining_quote:
                    #         quote_sents[-1] = remaining_quote
                    #         filler_intro[0] = mixed_filler
                    #     else:
                    #         quote_sents.pop()  
                    #         filler_intro[0] = mixed_filler



            elif quote_filler_mix_bool == False and len(quotes) == 1:
                count_single_quote_natural += 1
            


            # legg til quote linjer i listen
            quotes_splits.append(quote_sents)
            all_sentences.extend(quote_sents)
            


            #Vis det er flere enn en quote legg til random fillers mellom dem
            if i < len(quotes) - 1:
                n_fillers_mid = random.randint(1, 5)
                fillers_mid = sample_unique_fillers(n_fillers_mid, filler_sentences, used_filler_sentences)
                if fillers_mid is None:
                    return None
                all_sentences.extend(fillers_mid)

        filler_before = sample_unique_fillers(random.randint(1, 10), filler_sentences, used_filler_sentences)
        if filler_before is None or len(filler_before) == 0:
            log(f"[Warning] Ran out of filler sentences for filler_before at idx {idx}")

            return None
        filler_after = sample_unique_fillers(random.randint(1, 10), filler_sentences, used_filler_sentences)
        if filler_after is None or len(filler_after) == 0:
            log(f"[Warning] Ran out of filler sentences for filler_before at idx {idx}")
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

      
        if len(quotes) == 1 and quote_filler_mix_bool:
            parts = []
            print(f"quotes: {quotes}\n")
            print(f"matched_quote_lines: {matched_quote_lines}\n")
            quote = quotes[0]
            print(f"Quote: {quote}\n")
            quote_words = quote.split()  # splitt sitatet i ord
            print(f"Quote words splitted: {quote_words}\n")
            
            # Rens sitatordene for punktum, komma osv. og gjør lowercase for sammenligning
            quote_words_cleaned = [re.sub(r"[^\w\s’']", '', word).lower() for word in quote_words]
            print(f"Quote words cleaned: {quote_words_cleaned}\n")

            i = 0  # indeks for sitatord
            new_matched_quote_lines = []

            for start_t, end_t, text in matched_quote_lines:
                original_words = text.strip().split()
                cleaned_words = [re.sub(r"[^\w\s’']", '', w).lower() for w in original_words]

                matched_words = []
                for orig_word, clean_word in zip(original_words, cleaned_words):
                    if i < len(quote_words_cleaned) and clean_word == quote_words_cleaned[i]:
                        # Ta med ordet fra sitatet med original casing (fra quote_words)
                        matched_words.append(quote_words[i])
                        i += 1
                    else:
                        # Ord matcher ikke neste ord i sitatet, hopp over det
                        pass
                
                if matched_words:
                    new_matched_quote_lines.append((start_t, end_t, ' '.join(matched_words)))
                
                if i >= len(quote_words_cleaned):
                    # Hele sitatet er funnet, stopp
                    break

            # Bygg opp output med tidsstempler og teksten som matcher
            for start_t, end_t, text in new_matched_quote_lines:
                if start_t is not None and end_t is not None:
                    parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                else:
                    parts.append(text)

            full_quote_with_timestamps = " ".join(parts).replace('"', '\\"')
            print(f"parts: {parts}")
            print(f"full quote with timestamps: {full_quote_with_timestamps}")

            label_text = (
                'Thought: I found 1 standalone motivational passage that meet the criteria, so I’m saving it.\n' +
                '<code>\n'
                f'SaveMotivationalText(text="{full_quote_with_timestamps}", text_file=text_file)\n'
                'final_answer("Im done analyzing the chunk")\n'
                '</code>'
            )
            
        elif len(quotes) == 1 and not quote_filler_mix_bool:
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
            Thought = f"I found {len(save_calls)} standalone motivational passage{'s' if len(save_calls) > 1 else ''} that meet the criteria, so I’m saving {'them' if len(save_calls)>1 else 'it'}."
            label_text = (
                f"Thought: {Thought}\n"
                "<code>\n" + "\n".join(save_calls) + "\n"
                'final_answer("Im done analyzing the chunk")\n</code>'
            )

    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": instruction.strip() + "\n\n" + chunk_text.strip()},
            {"role": "assistant", "content": label_text.strip()}
        ]
    }
def load_quote_dataset():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\Quote_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quotes = [entry["quote"].strip() for entry in data if entry["quote"].strip()]
    return quotes

def main():
    used_filler_sentences = set()

    filler_sentences = load_filler_sentences()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)

    quotes_list = load_quote_dataset()
    dataset = [{"quote": q} for q in quotes_list]



    quotes_list = [x['quote'] for x in dataset]

    train_data = []
    train_ds = dataset

        


    for idx, example in enumerate(train_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, quotes_list)
        if processed is None:
            break
        train_data.append(processed)



    train_data_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\train.jsonl"
    with open(train_data_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


    print(f"Laget totalt: [{Count_double_quotes}] dobble quote eksempler.")
    print(f"Laget totalt: [{count_single_Quote_mix}] single quote/filler mix eksempler")
    print(f"Laget totalt:  [{count_only_filler}] Ingen sitater eksempler")
    print(f"Laget totalt: [{count_single_quote_natural}] single quote uten mix eksempler")

   # from trunacte import trunacte_dataset
   # trunacte_dataset(train_data_path)
    print("DONE!")

    log("done")

if __name__ == "__main__":
    main()
