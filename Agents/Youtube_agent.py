
import yaml
from utility.log import log
import torch
import gc
import datetime


def get_automatic_data_from_agent(model,input_video):
    """
    This function is designed to generate SEO metadata for a given input video using an smolagents step by step based approach.
    The agent utilizes various tools to extract audio, transcribe it, analyze trending YouTube videos, and perform web searches to synthesize high-performance metadata.
    The function returns the generated title, description, hashtags, tags, categoryId, and publishAt date for the video.
    """
    from smolagents import CodeAgent,FinalAnswerTool,GoogleSearchTool,PythonInterpreterTool,VisitWebpageTool,DuckDuckGoSearchTool
    from Agents.utility.Agent_tools import (
        ExtractAudioFromVideo,
        Fetch_top_trending_youtube_videos,
        SpeechToTextTool_viral_agent,

        get_upload_counts_by_date,
    )
    import utility.Global_state as Global_state

    from utility.upload_Socialmedia import Get_latestUpload_date
    YT_channel = Global_state.get_current_yt_channel()
    previous_publishAt = Get_latestUpload_date(YT_channel)
    import utility.Global_state as Global_state
    with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Youtube Agent\system_prompt.yaml"), 'r') as stream:
                Manager_Agent_prompt_templates = yaml.safe_load(stream)

    final_answer = FinalAnswerTool()
    fetch_youtube_video_information = Fetch_top_trending_youtube_videos
    PythonInterpeter = PythonInterpreterTool()
    DuckDuckGoSearch = DuckDuckGoSearchTool()
    Google_Websearch = GoogleSearchTool()


    manager_agent  = CodeAgent(
        model=model,
        tools=[
                final_answer,
                fetch_youtube_video_information,
                get_upload_counts_by_date,
                ],
        max_steps=5,
        verbosity_level=2,
        prompt_templates=Manager_Agent_prompt_templates,
        additional_authorized_imports=['datetime'],
    )

    emojies = "⚡🧠"
    context_vars = {
            "emojies": emojies,
            "previous_publishAt": previous_publishAt,
            'Already_uploaded_path': f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.json",
            'current_video_name': Global_state.get_current_videourl(),
            'present_datetime': datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

    audio_path =  ExtractAudioFromVideo(video_path=input_video)
    log(f"[Youtube_agent] Extracted audio path: {audio_path}")
    tool = SpeechToTextTool_viral_agent()
    tool.setup()
    transcript_text = tool.forward({"audio": audio_path})

    user_task = f"""
    Your task/goal is to generate a complete, viral-optimzized YouTube metadata package for the following `input_video`:
    Niche/genre: Motivational
    Transcript: {transcript_text}

    Respond with ONLY a valid JSON object via the `final_answer` tool containing the keys: `title`, `description`, `tags`, `hashtags`, `categoryId`, `publishAt`.
    All keys must be tightly aligned with the input video's niche and intent inferred from the transcript and research and optimized for maximum reach and virality.
    """

    try:
        Response = manager_agent.run(
            task=user_task,
            additional_args=context_vars,
        )
    except Exception as e:
            log(f"Error during agent run: {str(e)}")
    print(Response)
    import json
    try:
            data = Response if isinstance(Response, dict) else json.loads(Response)
    except Exception as e:
            raise ValueError(f"agent output is not valid json...!")
    torch.cuda.empty_cache()
    gc.collect()

    return (
    data.get("title"),
    data.get("description"),
    data.get("hashtags"),
    data.get("tags"),
    data.get("categoryId"),
    data.get("publishAt"),
)

















def get_automatic_data_from_agent_montage(model,input_video,YT_channel):
    """
    This function is designed to generate SEO metadata Montage (short-clips) for a given input video using an agent-based approach.
    The agent utilizes various tools to extract audio, transcribe it, analyze trending YouTube videos, and perform web searches to synthesize high-performance metadata.
    The function returns the generated title, description, hashtags, tags, categoryId, and publishAt date for the video.
    """
    from smolagents import CodeAgent,FinalAnswerTool,GoogleSearchTool,PythonInterpreterTool,VisitWebpageTool
    from Agents.utility.Agent_tools import ExtractAudioFromVideo,Fetch_top_trending_youtube_videos,SpeechToTextTool_viral_agent
    from utility.upload_Socialmedia import Get_latestUpload_date
    import torch

    with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Youtube Agent\system_prompt.yaml"), 'r') as stream:
                Manager_Agent_prompt_templates = yaml.safe_load(stream)
    previous_publishAt = Get_latestUpload_date(YT_channel)
    final_answer = FinalAnswerTool()
    Extract_audio = ExtractAudioFromVideo
    fetch_youtube_video_information = Fetch_top_trending_youtube_videos
    Transcriber = SpeechToTextTool_viral_agent()
    PythonInterpeter = PythonInterpreterTool()

    Google_Websearch = GoogleSearchTool()


    manager_agent  = CodeAgent(
        model=model,
        tools=[
                final_answer,
                Extract_audio,
                Transcriber,
                fetch_youtube_video_information,
                PythonInterpeter,
                Google_Websearch,
                VisitWebpageTool(),
                ],
        max_steps=5,
        verbosity_level=1,
        prompt_templates=Manager_Agent_prompt_templates,
        additional_authorized_imports=['datetime'],
    )


    emojies = "⚡🧠"
    context_vars = {
        "input_video": input_video,
        "emojies": emojies,
        "previous_publishAt": previous_publishAt,
        'present_datetime': datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    user_task = """
        Your goal is to generate the ultimate SEO metadata for the provided `input_video`.
        - **CRITICAL**: Do NOT copy-paste raw data from tool outputs. You must SYNTHESIZE your research to craft unique, high-performance metadata.
        - **ALIGNMENT**: Ensure `title`, `description`, `hashtags`, `tags`, `categoryId`, and `publishAt` are tightly optimized for the specific niche identified in the transcript and aligned with current viral trends.

        1. **Content Extraction & Analysis**:
            - First, use `ExtractAudioFromVideo` and `SpeechToTextTool_viral_agent` to get the full transcript.
            - Deeply analyze the transcript to identify the specific motivational niche (e.g., "Stoic Discipline", "Overcoming Heartbreak", "Financial Hustle").

        2. **Strategic Research (YouTube)**:
            - Use `Fetch_top_trending_youtube_videos` to find high-performing competitors with similar content. Analyze their popular video's metadata.

        3. **Metadata Generation**:
            - Synthesize all findings to create a valid JSON object with: `title`, `description`, `tags`, `hashtags`, `categoryId`, and `publishAt`.
            - The `title` must be a high-CTR hook derived from your research.
            - The `description` must be SEO-optimized with relevant keywords and a compelling summary.
            - The `tags` must include trending and relevant keywords specific to the video's niche.
            - The `hashtags` must be relevant and trending within the specific motivational niche.
            - The `categoryId` must accurately reflect the video's content (e.g., "Mot
            - The `publishAt` must follow the scheduling logic relative to `previous_publishAt` and `Already_uploaded_list`.

        Return ONLY the valid JSON object using the `final_answer` tool
        """
    try:
        Response = manager_agent.run(
            task=user_task,
            additional_args=context_vars
        )
    except Exception as e:
            log(f"Error during agent run: {str(e)}")
    print(Response)
    import json
    try:
            data = Response if isinstance(Response, dict) else json.loads(Response)
    except Exception as e:
            raise ValueError(f"agent output is not valid json...!")
    torch.cuda.empty_cache()
    gc.collect()

    return (
    data.get("title"),
    data.get("description"),
    data.get("hashtags"),
    data.get("tags"),
    data.get("categoryId"),
    data.get("publishAt"),
)


