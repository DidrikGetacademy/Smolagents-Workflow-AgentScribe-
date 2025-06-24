import os
import sys
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from dotenv import load_dotenv
from smolagents import CodeAgent, FinalAnswerTool,  DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool, TransformersModel,VLLMModel, SpeechToTextTool,PythonInterpreterTool,SpeechToTextToolCPU_VIDEOENCHANCERPROGRAM
from Agents_tools import ExtractAudioFromVideo, Fetch_top_trending_youtube_videos,Read_transcript
import gc
import torch
load_dotenv()
import yaml
already_uploaded_videos = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\already_uploaded.txt"                
from log import log
# Scopes required to upload video
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def get_authenticated_service():
    creds = None
    if os.path.exists("./youtube_token.pickle"):
        with open("./youtube_token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.getenv("YOUTUBE_API_KEY"), SCOPES)
            creds = flow.run_console()
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)

def upload_video(model,file_path, category_id="22", privacy_status="private"):
    youtube = get_authenticated_service()
    try:
      title, description, tags, time_date = get_automatic_data_from_agent(model,file_path)
    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent]")

         
    with open(already_uploaded_videos,"a", encoding="UTF-8") as w:
         w.write(time_date + "\n")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "publishAt": time_date
        }
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    print(f"Upload complete! Video ID: {response['id']}")



def get_automatic_data_from_agent(model,input_video):
        #Agent Prompts
        with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\Viral_prompt_template.yaml"), 'r') as stream:
                    Manager_Agent_prompt_templates = yaml.safe_load(stream)


        #Tool initalization
        final_answer = FinalAnswerTool()
        web_search = DuckDuckGoSearchTool()
        Extract_audio = ExtractAudioFromVideo
        fetch_youtube_video_information = Fetch_top_trending_youtube_videos
        Transcriber = SpeechToTextToolCPU_VIDEOENCHANCERPROGRAM()
        PythonInterpeter = PythonInterpreterTool()
        Visit_WebPage = VisitWebpageTool()



        
        manager_agent  = CodeAgent(
            model=model,
            tools=[
                  web_search,
                  final_answer,
                  Visit_WebPage,
                  Extract_audio,
                  Transcriber,
                  PythonInterpeter,
                  Visit_WebPage,
                  fetch_youtube_video_information,
                  ], 
            max_steps=6,
            verbosity_level=4,
            prompt_templates=Manager_Agent_prompt_templates,
            add_base_tools=True,
            stream_outputs=True,
            additional_authorized_imports=['datetime']
            
        )


        context_vars = {
               "video_path": input_video,
               "already_uploaded":already_uploaded_videos, 
             
            }       
        user_task = (
            "Generate a JSON object containing the following fields for a YouTube video: "
            "`title`, `description`, `tags`, and `time_date`. "
            "The goal is to create content that has high viral potential by leveraging current trends "
            "and analyzing successful videos in the same category. Include a unique message that highlights "
            "key insights, secret strategies, or specific elements that contributed to the virality of similar content. "
            "This unique message should help the video stand out and perform exceptionally well. "
            "Your final response must be a valid JSON object with ONLY these exact keys: "
            "`title`, `description`, `tags`, and `time_date`. "
            "`time_date` should be in RFC 3339 UTC format (e.g., '2025-06-23T14:30:00Z'). "
            "Do NOT include any other text or fields outside the JSON."
        )

        Response = manager_agent.run(
            task=user_task,
            additional_args=context_vars
        )
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
        data.get("tags"),
        data.get("time_date"),
    )




if __name__ == "__main__":
    from smolagents import TransformersModel
    model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Safetensor_models\Ministral-8B-Instruct-2410",
            load_in_4bit=True,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            max_new_tokens=5000,
    )
    upload_video(model=model,file_path=r"c:\Users\didri\Videos\Timeline 1.mov")
