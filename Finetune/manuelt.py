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
[06:51s - 07:00s] product that he sold and the guarantee was this He said all you have to do you can get all\n
[07:01s - 07:10s] your money back if you just open up a Google doc and say what you're going to do every\n
[07:11s - 07:20s] day and at the end of the day you're going to say what you did That's\n
[07:21s - 07:30s] it And you just do that for six weeks That's all you have to do And if you do \n
[07:31s - 07:40s] that and you dont get the the results Ill you know Ill give your money back And what's crazy about that is when he was talking about it he said \n
[07:41s - 07:50s] Do you know what the completion rate of twice a day doing something is he's like it's like less than 1% \n
[07:51s - 08:00s] He's like so I can happily do it because it seems so simple And so I think we we we underplay how simple success \n
[08:01s - 08:10s] is and extrapolate an expectation of how easy it must be from that And then we're disappointed \n
[08:11s - 08:20s] or dissatisfied when it doesn't meet those expectations Like it is harder than we expect but also the rewards may be also greater than we expect James Clear \n
[08:21s - 08:30s] has this unbelievable insight It doesn't make sense to continue wanting something if you're not willing \n
[08:31s - 08:40s] to do what it takes to get it If you don't want to live the lifestyle then release yourself from the desire\n
[08:41s - 08:50s] To crave the result but not the process is to guarantee disappointment\n
[08:51s - 09:00s] Yeah Holy Super true I mean um I think Naval quoted this blog a long time ago but desire\n
[09:01s - 09:10s] is a contract you make with yourself to be unhappy until\n
[09:11s - 09:20s] you get what you want And if you never \n
[09:21s - 09:30s] if you basically know that you're never \n
[09:31s - 09:40s] willing to put the work in to get\n
[09:41s - 09:50s] the thing Yep That's an\n
"""

quote1 = """
[07:31s - 07:40s] he said 
[07:41s - 07:50s] Do you know what the completion rate of twice a day doing something is he's like it's like less than 1% 
[07:51s - 08:00s] He's like so I can happily do it because it seems so simple And so I think we we we underplay how simple success 
[08:01s - 08:10s] is and extrapolate an expectation of how easy it must be from that And then we're disappointed 
[08:11s - 08:20s] or dissatisfied when it doesn't meet those expectations Like it is harder than we expect but also the rewards may be also greater than we expect
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


