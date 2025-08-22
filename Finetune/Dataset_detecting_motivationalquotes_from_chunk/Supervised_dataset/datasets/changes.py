import json

# Ny system_prompt du ønsker å bruke
new_system_prompt = """
  [Criteria for a Valid Motivational Short]:
    - Expresses a complete, self-contained thought or message that stands alone without needing prior context.
    - Conveys a clear, positive, and uplifting message, Inspirational passages, self-contained statements, or anecdotes that offer encouragement, guidance, or motivation. They may promote personal growth, resilience, reflection, or positive action, often using memorable insights, contrasts, or relatable experiences to foster a constructive mindset, perseverance, and self-improvement.
    - Is concise and punchy, suitable for a short motivational video.
    - Provides enough context and clarity so the overall intent, lesson, or insight is immediately understandable and does not confuse the listener.

  [Reasons to Reject a Motivational Short]:
    -The message is incomplete, unclear, or requires prior context to make sense.
    -Lacks a positive or empowering takeaway, or is neutral/negative.
    -Is too long, rambling, or not suitable for a short video format.
    -Relies heavily on personal anecdotes, Avoid vagueness that aren’t widely relatable or incomplete ideas that could confuse a listener.

  [Your OUTPUT structure in 'Thought' sequence should only be 3 parts]:
    Part 1: Identify text block (NUM) + first timestamp from full text between ===START_TEXT=== & ===END_TEXT===
    Part 2: Reason: Self-reflecting question (Is this a complete and standalone motivational message suitable for a motivational shorts video?)
         - This question decides if you reject the text or consider it valid for creating a video.
         - The text must meet most of the criteria, but most importantly, it must be self-contained and not depend on prior context. If made into a video, this text should not confuse a listener.
    Part 3: Decision (valid or reject) – based on the reflection.

    Tool output in '<code>':
      create_motivationalshort("") # for those that are valid
      Delete_rejected_line("===START_TEXT=== ===END_TEXT===") # for those that are rejected

    NOTE: You must remember that if there are only rejected texts, you should not include the create_motivationalshort tool in the <code> block. Likewise, if there are only valid texts, do not include Delete_rejected_line if nothing was rejected

    Task: "Analyze all the textblocks. You must Reject those that are not valid for a motivational short using `Delete_rejected_line`. You must Approve those that are valid by passing them into `create_motivationalshort`. Here is the text to analyze: ..."

    Thought:
    * Textblock 1: [timestamps]
      - Check: Is this a complete and standalone motivational message?
      - Reason: <your reasoning>
      - Decision: valid/reject

    * Textblock 2: [timestamps]
      - Check: Is this a complete and standalone motivational message?
      - Reason: <your reasoning>
      - Decision: valid/reject

    <code>:
    create_motivationalshort("...")
    Delete_rejected_line("===START_TEXT===...===END_TEXT===")
    final_answer("im done")
    </code>
"""

# Les inn original train.jsonl
input_file = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\verify_agent_dataset.jsonl"
output_file = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\new_verify_agent_dataset.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "a", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue
        obj = json.loads(line)

        # Antar at første melding alltid er system_prompt
        if obj["messages"][0]["role"] == "system":
            obj["messages"][0]["content"] = new_system_prompt.strip()

        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ Ferdig! Lagret i", output_file)
