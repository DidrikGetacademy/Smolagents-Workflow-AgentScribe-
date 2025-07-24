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

instruction = """ Here is the chunk you will analyze:\n

    """

chunk_text = """
[00:00s - 00:30s] you know man that we've been taught that failing is final. But the truth is, if you're doing anything meaningful
[00:31s - 01:00s] if you're building something, taking a risk, creating a life that’s true to you — you’re going to mess up. And that’s not just okay,
[01:01s - 01:30s] it’s necessary.
[01:31s - 02:00s] The people we admire the most?
[02:01s - 02:30s] They’ve all failed.
[02:31s - 03:00s] Repeatedly.
[03:01s - 03:30s] And what makes them different isn’t that they avoided failure,
[03:31s - 04:00s] it’s that they didn’t let it stop them. They learned, they adapted, 
[04:01s - 04:30s] and they kept going. the mind can be under alot of pressure when you work day in and day out.
[04:31s - 05:00s] yea people need to be careful
[05:01s - 05:30s] alot of stress can make the mind shutdown
[05:31s - 06:00s] yea people forget that
[06:01s - 06:30s] but did you know that last night 

"""
# we've been taught that failing is final. But the truth is, if you're doing anything meaningful — if you're building something, taking a risk, creating a life that’s true to you — you’re going to mess up. And that’s not just okay, it’s necessary.
# The people we admire the most? They’ve all failed. Repeatedly. And what makes them different isn’t that they avoided failure, it’s that they didn’t let it stop them. They learned, they adapted, and they kept going.

quote1 = """
[00:00s - 00:30s] we've been taught that failing is final. But the truth is, if you're doing anything meaningful
[00:31s - 01:00s] if you're building something, taking a risk, creating a life that’s true to you — you’re going to mess up. And that’s not just okay,
[01:01s - 01:30s] it’s necessary.
[01:31s - 02:00s] The people we admire the most?
[02:01s - 02:30s] They’ve all failed.
[02:31s - 03:00s] Repeatedly.
[03:01s - 03:30s] And what makes them different isn’t that they avoided failure,
[03:31s - 04:00s] it’s that they didn’t let it stop them. They learned, they adapted, 
[04:01s - 04:30s] and they kept going.
"""

quote2 = """"""

quote3 = """"""

label_text = "Thought: I found 1 standalone motivational passages that meet the criteria, so I’m saving them.\n<code>\n"

if quote1:
    label_text += f'SaveMotivationalText(text="{quote1.strip()}", text_file=text_file)\n'
if quote2:
    label_text += f'SaveMotivationalText(text="{quote2.strip()}", text_file=text_file)\n'
if quote3:
    label_text += f'SaveMotivationalText(text="{quote3.strip()}", text_file=text_file)\n'

label_text += 'final_answer("Im done analyzing the chunk")\n</code>'

def create_manual_exsamples():
        

    return {
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": instruction.strip() + "\n\n" + chunk_text.strip()},
                {"role": "assistant", "content": label_text.strip()}
            ]
        }

messages = create_manual_exsamples()
import json 
with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")


