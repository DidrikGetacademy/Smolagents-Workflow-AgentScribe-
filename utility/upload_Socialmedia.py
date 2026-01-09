import os
import sys
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from dotenv import load_dotenv
import gc
import json
from googleapiclient.errors import HttpError
import torch
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
from utility.log import log
import utility.Global_state as Global_state





def upload_tiktok_Instagram_API(hashtags,description,title,video_path,YT_channel):
    import requests
    if YT_channel == "LR_Youtube":
        Channel_name = "LR_tiktok_instagram_api_key"
    elif YT_channel == "LRS_Youtube":
        Channel_name = "LRS_tiktok_instagram_api_key"
    elif YT_channel == "MR_Youtube":
        Channel_name = "MR_tiktok_instagram_api_key"
    elif YT_channel == "LM_Youtube":
        Channel_name = "LM_tiktok_instagram_api_key"

    url = "https://api.ayrshare.com/api/post"
    api_key = os.getenv(Channel_name)
    log(f"api_key used for tiktok/instagram upload: {api_key} - channel_name: {Channel_name}")
    PostHead = ""
    title += description
    PostHead += title
    schedule_url = "https://api.ayrshare.com/api/auto-schedule/set"
    schedule_data = {
        "title": "AfterNoon Schedule",
        "schedule": ["09:00Z", "14:00Z"],
        "daysOfWeek": ["Monday", "Wednesday", "Friday"],
        "timezone": "UTC"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
            response = requests.post(schedule_url, headers=headers, json=schedule_data)
            log(f"Schedule response: {response.json()}")
    except Exception as e:
         log(f"Error during scheduling: {str(e)}")


    data = {
        "post": PostHead,
        "platforms": ["tiktok", "instagramApi"],
        "mediaUrls": video_path,
        "hashtags": hashtags,
        "isVideo": True,
        "autoSchedule": {"schedule": True, "title": schedule_data["title"]}


    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        log(f"Video uploaded successfully to TikTok and Instagram! response: {response.json()}")
    else:
        log(f"Failed to upload video. Status code: {response.status_code}, Response: {response.text}")


def Populate_Already_Uploaded(title, description, tags, categoryId, publishAt, YT_channel, video_url,subtitle_text,error,background_audio_,song_name,video_duration=None):
    history_path = os.path.join("Video_clips", "Youtube_Upload_folder", YT_channel, "already_uploaded.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as r:
                history = json.load(r)
        except Exception as e:
            log(f"Failed to load existing upload history: {str(e)}")

    subtitle_text_list = []
    for text in subtitle_text:
        subtitle_text_list.append(text)

    entry = {
        "title": title,
        "description": description,
        "tags": tags,
        "categoryId": categoryId,
        "publishAt": publishAt,
        "youtube_channel": YT_channel,
        "current_video_name": video_url,
        "subtitle_text": subtitle_text_list,
        "video_duration": video_duration,
        "background_audio": background_audio_,
        "song_name": song_name,
        "error": error,
    }
    history.append(entry)

    try:
        with open(history_path, "w", encoding="utf-8") as w:
            json.dump(history, w, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Failed to write upload history json: {str(e)}")


SCOPES = ["https://www.googleapis.com/auth/youtube"]
def get_single_playlist_id(youtube):
    try:
          request = youtube.playlists().list(
               part="id",
               mine=True,
               maxResults=1
          )
          response = request.execute()

          items = response.get("items", [])
          log(f"response from (get_single_playlist_id): {items}")
          if items:
               item_returned = items[0]["id"]
               log(f"returned playlist_id: {item_returned}")
               return items[0]["id"]

          else:
               log("No playlist found")
               return
    except HttpError as e:
        log(f"Error retrieving playlist_id : {e}")



def get_authenticated_service(YT_channel):
    if YT_channel == "LR_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Secrets\LR_Youtube\client_secret_849553263621-grvsfihl7lkocrt0qgs37iipuv5lbpkl.apps.googleusercontent.com.json"
    elif YT_channel == "LRS_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Secrets\LRS_Youtube\client_secret_2_631618294192-2maahe6qccmd7naepvd54i01ur33h1js.apps.googleusercontent.com.json"
    elif YT_channel == "MR_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Secrets\MR_Youtube\client_secret_2_69920788682-r76oarkr2q94p99k76svqacvtv66b24a.apps.googleusercontent.com.json"
    elif YT_channel == "LM_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Secrets\LM_Youtube\client_secret_2_460945830865-u9g1codhuh14ht71rsdcdq7ip6dgb9cq.apps.googleusercontent.com.json"
    elif YT_channel == "MA_Youtube":
            Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Secrets\MA_Youtube\client_secret_813947315190-c247pb2s274uh22sbite599gabcp87vq.apps.googleusercontent.com.json"
    else:
         raise ValueError(f"Error No {YT_channel} exists.")

    print(Client_secret)

    base_dir = os.path.dirname(Client_secret)
    pricle_path =  os.path.join(base_dir, 'youtube_token.pickle')
    creds = None
    if os.path.exists(pricle_path):
        with open(pricle_path, "rb") as token:
            creds = pickle.load(token)
            log(f"Loaded pickle: {creds}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                Client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(pricle_path, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)



def Get_latestUpload_date(YT_channel: str):
    """Retrieves the latest video from the authenticated YouTube channel, including scheduled videos,
    and provides the publish date for you to know when the last video was or will be published.

    This function uses OAuth authentication to access the authenticated user's YouTube channel.
    It fetches the uploads playlist, retrieves the most recent video, and returns the video title
    along with its `publishAt` timestamp, which is the key information for scheduling or planning  the next video upload

    Args:
        YT_channel: The identifier or name of the authenticated YouTube channel. This is used
                    to locate the appropriate OAuth credentials for the channel.

    Returns:
        A formatted string containing:
        - The title of the latest video
        - The `publishAt` date/time (scheduled or actual publish date)
        This is the critical information you should use to track when the last video was/will be published.
        If no videos are found, returns a failure message.

    Example:
        >>> Get_latestUpload_date(YT_channel=YT_Channel)
        'Latest Video uploaded for the current Youtube Channel is:
            Title: How to Learn Faster
         \n PublishedAt: 2025-09-06T18:00:00Z'
    """

    youtube = get_authenticated_service(YT_channel)
    channel_request = youtube.channels().list(
         part="contentDetails",
         mine=True
    )
    channel_response = channel_request.execute()
    uploads_playlist = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


    playlist_request = youtube.playlistItems().list(
         part="snippet",
         playlistId=uploads_playlist,
         maxResults=2
    )

    playlist_response = playlist_request.execute()
    items = playlist_response.get("items",[])
    if not items:
         return "Failed getting latest publishAt"

    videos = [] # liste av videos
    for item in items: # for hver item i listen fra response
         video_id = item["snippet"]["resourceId"]["videoId"]

         video_request = youtube.videos().list(
              part="snippet,status",
              id=video_id
         )

         video_response = video_request.execute()
         video_info = video_response["items"][0]

         publish_time = video_info["status"].get(
             "publishAt",
              video_info["snippet"]["publishedAt"]
              )
         videos.append(publish_time)
    log(f"videos: {videos}")
    return videos



def upload_video(model,file_path,YT_channel,subtitle_text=None,background_audio_=None,song_name=None,video_duration=None):
    try:
      log(f"Autenchiating now..: youtube_channel: {YT_channel}")
      youtube = get_authenticated_service(YT_channel)
      log(f"auth success")
    except Exception as e:
         log(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      from Agents.Youtube_agent import get_automatic_data_from_agent
      title, _description, hashtags, tags, categoryId, publishAt = get_automatic_data_from_agent(model,file_path)
      log(f"[get_authenticated_service]: title: {title},\n description: {_description},\n  hashtags: {hashtags},\n tags: {tags},\n categoryId: {categoryId},\n publishAt: {publishAt}\n")

    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return


    for tag in hashtags:
         _description += f"{tag} "

    import utility.Global_state as Global_state
    video_url = Global_state.get_current_videourl()

    body = {
        "snippet": {
            "title": title,
            "description": _description,
            "tags": tags,
            "categoryId": categoryId,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publishAt,
            "selfDeclaredMadeForKids": False
        }
    }
    try:
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
                log(f"Upload progress: {int(status.progress() * 100)}%")
        log(f"Upload complete! Video ID: {response['id']}")

        playlist_id = get_single_playlist_id(youtube)
        play_list_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": response['id']
                    },
                    "position": 0,
                }
            }
        try:
            error = False
            playlist_response = youtube.playlistItems().insert(
                part="snippet",
                body=play_list_body
            ).execute()
            log(f"Video added to playlist {playlist_id}: {playlist_response['id']}")
            try:
                Populate_Already_Uploaded(title,_description,tags,categoryId,publishAt,YT_channel,video_url,subtitle_text,error,background_audio_,song_name=song_name,video_duration=video_duration)
            except Exception as e:
                    log(f"error writing to populate already uploaded: {str(e)}")
            truncated_audio_path = Global_state.get_current_truncated_audio_path()
            os.remove(truncated_audio_path)
        except Exception as e:
            error = True
        return YT_channel
        # try:
        #      upload_tiktok_Instagram_API(hashtags=hashtags,description=_description,title=title,video_path=file_path,YT_channel=YT_channel)
        # except Exception as e:
        #     log(f"Error during uploading to Tiktok/Instagram: {str(e)}")
    except Exception as e:
         log(f"Error during uploading: {str(e)}")





def upload_MontageClip(model,file_path,YT_channel,subtitle_text=None,background_audio_=None):
    try:
      log(f"Autenchiating now..: youtube_channel: {YT_channel}")
      youtube = get_authenticated_service(YT_channel)
      log(f"auth success")
    except Exception as e:
         log(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      from Agents.Youtube_agent import get_automatic_data_from_agent_montage
      title, _description, hashtags, tags, categoryId, publishAt = get_automatic_data_from_agent_montage(model,file_path,YT_channel)
      log(f"[get_authenticated_service]: title: {title}, descriptino: {_description}, tags: {tags}, publishAt: {publishAt}")

    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return


    for tag in hashtags:
         _description += tag

    video_url = None

    body = {
        "snippet": {
            "title": title,
            "description": _description,
            "tags": tags,
            "categoryId": categoryId,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publishAt,
            "selfDeclaredMadeForKids": False
        }
    }
    try:
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
                log(f"Upload progress: {int(status.progress() * 100)}%")
        log(f"Upload complete! Video ID: {response['id']}")

        playlist_id = get_single_playlist_id(youtube)
        play_list_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": response['id']
                    },
                    "position": 0,
                }
            }
        try:
            error = False
            playlist_response = youtube.playlistItems().insert(
                part="snippet",
                body=play_list_body
            ).execute()
            log(f"Video added to playlist {playlist_id}: {playlist_response['id']}")

            Populate_Already_Uploaded(title,_description,tags,categoryId,publishAt,YT_channel,video_url,subtitle_text,error,background_audio_)
        except Exception as e:
            error = True
        # try:
        #      upload_tiktok_Instagram_API(hashtags=hashtags,description=_description,title=title,video_path=file_path,YT_channel=YT_channel)
        # except Exception as e:
        #     log(f"Error during uploading to Tiktok/Instagram: {str(e)}")
        #     log(f"error adding video to playlist {str(e)}")
    except Exception as e:
         log(f"Error during uploading: {str(e)}")



global test_log_list
test_log_list = []

if __name__ == "__main__":
     from smolagents import LiteLLMModel
     gc.collect()
     torch.cuda.empty_cache()
     Global_state.set_current_videourl(r"c:\Users\didri\Documents\Is Being Smart Worth the Depression？ - Alex O’Connor & Joe Folley (4K).mp4")
     Global_model = LiteLLMModel(model_id="gpt-5", reasoning_effort="minimal" ,api_key=OPENAI_APIKEY,max_tokens=20000)
     Global_state.set_current_yt_channel("MA_Youtube")
     upload_video(model=Global_model,file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\LM_Youtube\short11_rife.mp4",subtitle_text="",YT_channel="LRS_Youtube",background_audio_="lofi")
