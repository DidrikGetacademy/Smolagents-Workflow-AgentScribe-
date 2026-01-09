
from Agents.utility.Agent_tools import open_work_file,montage_short_creation_tool
from smolagents import  CodeAgent
import yaml
from utility.log import log


def Run_short_montage_agent():
    """Agent that creates motivational montage shorts from multiple video titles with snippets"""
    global Global_model
    from App import Reload_and_change_model
    Global_model =  Reload_and_change_model("gpt-5-high",message="Reloading model to -> gpt-5 before running [Montage_short_agent]")
    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Montage short Agent\system_prompt.yaml", "r", encoding="utf-8") as r:
         montage_agent_systemprompt = yaml.safe_load(r)

         Montage_task = """
            Your task is to create the best motivational montage shorts video with motivational snippets.

            Input:
                - text file with 3–8 video titles, each representing a video.
                - Each video title section contains  “motivational snippets” (quotes/excerpts).

            Task Goal:
                - Combine content from each video into a single, cohesive, and motivating script (for a short-form video).

            Workflow:
            1. Analyze snippets in each video with data returned from the `open_work_file` tool.
                - Read all snippets per file.
                - Understand the core message and tone (e.g., perseverance, growth, overcoming fear, discipline).
                - Consider merges or connection for a strong punshy motivational short montage.

            2. Select compatible content from each Video title
                - You may choose:
                    * A full snippet, OR
                    * One or more complete sentences from within a snippet (only if the sentence has more than 4 words).
                - Chosen content must naturally fit together.
                - Ensure smooth flow, so when played in sequence, it sounds like one motivational montage speech.
                - Avoid combinations that feel disjointed or contradictory.
                - Make sure that when the content from each video title is composed. It does not exceed the length of 30 seconds.

            3. Decide the sequence
                - Arrange chosen content from each Video Title in a logical order:
                    * Opening: something that hooks attention or sets the theme.
                    * Middle:  Text that provide a suitable context/flow with the opening and ending, The amount of middle parts can vary depending on the number of video titles provided.
                    * Ending: a powerful punchline, encouragement, or call-to-action.

            4. Generate the output
                - Deliver 5 `montage_short_creation_tool` tool calls, one for each YT_Channel.
                - Make sure it can be read aloud smoothly and feels like a finished motivational montage speech.

            """
    agent = CodeAgent(
         model=Global_model,
         verbosity_level=1,
         max_steps=5,
         prompt_templates=montage_agent_systemprompt,
         tools=[montage_short_creation_tool,open_work_file],
         use_structured_outputs_internally=True
        )
    additional_args = {
         "work_queue_folder": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder"
         }


    response = agent.run(task=Montage_task, additional_args=additional_args)
    log(f"Montage Agent response: {response}")

    del agent
