from smolagents import  CodeAgent,TransformersModel,FinalAnswerTool
import torch
import yaml

chunk_test1 = """
[2210.45s - 2215.81s] Because I've had to go through it myself. I often say, you know, heaven is a place on earth, but you have to go through  
[2215.82s - 2220.13s] the storm to truly appreciate the sunshine.  
[2220.14s - 2224.00s] Not every day will be easy, but every step forward matters.  
[2224.01s - 2227.44s] People always told me, “The comeback is stronger than the setback.”  
[2227.45s - 2231.10s] Sometimes, you just need to breathe and trust that things will work out.  
[2231.11s - 2235.22s] I mean, there were days I didn't believe in myself, and that’s okay.  
[2235.23s - 2239.50s] My mentor used to say, “Success is built on a thousand quiet efforts.”  
[2239.51s - 2243.30s] And now I understand — growth doesn't always look like progress.
[2210.45s - 2215.81s] Because I've had to go through it myself. I often say, you know, heaven is a place on earth, but you have to go through  
[2215.82s - 2220.13s] the storm to truly appreciate the sunshine.  
[2220.14s - 2224.00s] Not every day will be easy, but every step forward matters.  
[2224.01s - 2227.44s] People always told me, “The comeback is stronger than the setback.”  
[2227.45s - 2231.10s] Sometimes, you just need to breathe and trust that things will work out.  
[2231.11s - 2235.22s] I mean, there were days I didn't believe in myself, and that’s okay.  
[2235.23s - 2239.50s] My mentor used to say, “Success is built on a thousand quiet efforts.”  
[2239.51s - 2243.30s] And now I understand — growth doesn't always look like progress.  
[2243.31s - 2246.20s] You know, we all have our own battles, and it's not always obvious.  
[2246.21s - 2249.50s] It's just that some lessons take longer to land than others.  
[2249.51s - 2252.00s] I kept going, even when I didn’t know what I was moving toward.
"""

chunk_test2 = """
[3000.00s - 3004.50s] Life has its ups and downs, but it’s how you respond that defines you.  
[3004.51s - 3008.75s] I remember hearing, “Your mindset shapes your reality.”  
[3008.76s - 3012.00s] Sometimes, the smallest step in the right direction ends up being the biggest step of your life.  
[3012.01s - 3015.20s] You might not see progress every day, but persistence wins in the long run.  
[3015.21s - 3018.55s] Honestly, some days I just feel stuck, and that’s part of the journey.  
[3018.56s - 3022.10s] A wise friend once said, “Doubt kills more dreams than failure ever will.”  
[3022.11s - 3025.80s] Not every piece of advice fits everyone — you have to find what works for you.  
[3025.81s - 3029.00s] The real strength is in getting back up after you fall.  
[3029.01s - 3032.50s] I guess, sometimes you just need to listen to your own voice instead of others’.  
[3032.51s - 3036.00s] Remember, “Success is not final, failure is not fatal: It is the courage to continue that counts.”  
[3036.01s - 3039.45s] Life isn’t a race; it’s a marathon with its own pace.  
"""


chunk_test3 = """

"""

Test_task = f"You are given a text chunk to carefully analyze for motivational or inspirational quotes, advice, or any meaningful content that conveys wisdom. Your goal is to identify text that could be used in a motivational short. Please analyze the chunk as a human reader would. The text chunk to analyze is: {chunk}"
Model = r"C:\Users\didri\Desktop\LLM-models\Qwen\Qwen_3B"
def test_model_reasoning_ability_on_Chunk(Model, chunk):
    Transformer_model = TransformersModel(
        model_id=Model,
        device_map="cuda",
        torch_dtype=torch.float16,
        load_in_8_bit=True,
        trust_remote_code=True
    )

    with open(r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\venv",  "r", encoding="utf-8") as file:
        Agent_prompt_template = yaml.safe_load(file)


    agent = CodeAgent(
        model=Transformer_model,
        max_steps=10,
        tools=[FinalAnswerTool],
        add_base_tools=True,
        prompt_templates=Agent_prompt_template
    )
    agent.run(task=Test_task)


    
