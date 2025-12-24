
from utility.Custom_Agent_Tools import extract_window_frames

def run_multi_Vision_model_agent(video_path , Additional_content,Global_model, already_uploaded_videos,start_time,end_time) -> str:
    """Agent that selects background music for the motivational shorts video
    Generates a detailed motivational summary of a podcast using both text and audiovisual inputs.
    The function leverages a Vision-Audio model to process the video and audio content,
    and a text summarization pipeline to produce a concise, motivational summary.

    Args:
        _text: The transcript or textual content of the podcast to summarize.
        video_path: Path to the video file that is being uploaded of the podcast, used to extract visual frames for context.
        audio_path: Path to the audio file that is being uploaded of the podcast, used for extracting audio features for context.

    Returns:
        A string containing a detailed summary of the podcast, highlighting motivational points, emotions,
        and actionable advice. If summarization fails, a message indicating the failure is returned.
    """
    from smolagents import CodeAgent,FinalAnswerTool
    from PIL import Image
    import yaml
    frames = extract_window_frames(video_path, start_time=start_time, end_time=end_time, max_frames=5)


    with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Agents\Prompt_templates\Background Music Agent\system_prompt.yaml", "r", encoding="utf-8") as r:
            prompt = yaml.safe_load(r)

    with open(already_uploaded_videos, "r", encoding="utf-8") as f:
            uploaded_videos_content = f.read()



    agent = CodeAgent(
        model=Global_model,
        add_base_tools=True,
        verbosity_level=1,
        prompt_templates=prompt,
        tools=[FinalAnswerTool()],
        additional_authorized_imports=['typing'],
        max_steps=3,
    )

    user_message = f"""Analyze the following inputs to select the best background music:
    - Podcast Transcript: ({Additional_content["Transcript"]}) -  (Use this to identify the story, key motivational points, and overall arc.)

    - Detected Speech Emotion: ({Additional_content["emotion"]})  (Raw tone from audio model; integrate but prioritize motivational intent if conflicting.)

    - Video Frames and Audio: Observe all the  provided images for visual style, speaker energy, and pacing. Listen to the audio for vocal delivery, pauses, and intensity.

    - Candidate Background Tracks: ({Additional_content["videolist"]})  (List with metadata; select exactly one by its number.)

    - Previously Used Tracks/songs: \n({uploaded_videos_content})\n  (Avoid repeating any tracks already used in prior videos. the limit before you are allowed to reuse is 3 videos ago.)
    Follow the reasoning process step-by-step in your thinking, but output only the JSON with your final choice. Ensure the music enhances clarity, emotion, and motivation for the audience.

    -Respond ONLY with valid JSON  object inside your `final_answer` tool
        - Structure:  "path": "<exact file path from videolist>",  "song_name": "The name of the song choosen","reason": "<concise explanation, 1-3 sentences>", "editing_notes": "<optional suggestions, e.g., 'Fade in at 5s, duck volume during key speeches'>", "lut_path": <The path for choosen lut for the shorts video>"
        - The "path" field must match the "path" value from one of the tracks in the provided videolist exactly.
    """


    response =  agent.run(task=user_message, images=frames)

    return response




