
import os
import sys
import json
from datetime import datetime
from langgraph.graph import END, StateGraph, START
from operator import add
from typing import Dict, List, Optional, TypedDict,Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent,AgentState
import yt_dlp
from googleapiclient.discovery import build
from utility.log import log
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@tool
def Check_Already_used_videos(video_id: str, channel_name: str) -> bool:
    """
    A tool that checks if a YouTube video ID has already been used by reading from a JSON file.

    Args:
        video_id (str): The YouTube video ID to check.
        channel_name (str): The name of the channel.
    Returns:
        bool: True if the video ID has been used, False otherwise.
    """
    import json
    youtube_folder = os.path.join(REPO_ROOT, "Youtube", channel_name)
    json_file = os.path.join(youtube_folder, "used_videos.json")

    if not os.path.exists(json_file):
        return False

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            used_videos = json.load(f)

        for video in used_videos:
            if video.get("video_id") == video_id:
                return True
    except Exception as e:
        log(f"Error reading used_videos.json: {e}")
        return False

    return False


@tool
def check_video_source_exists(video_name: str) -> str:
    """
    Checks if a video source name exists in the completed videos database.
    This tool searches the Videosources_completed.json file to see if a specific video has already been used.

    Args:
        video_name (str): The exact name of the video to check (e.g., "Motivational Speech.mp4")

    Returns:
        str: A message indicating whether the video exists and its metadata if found.
    """
    json_path = r"c:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Videosources_completed.json"

    try:
        if not os.path.exists(json_path):
            return f"Database file not found. Video '{video_name}' does not exist in database."

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = video_name.strip()

        if video_name in data:
            metadata = data[video_name]
            used_date = metadata.get("used_date", "unknown")
            clips_created = metadata.get("clips_created", 0)
            return f"Video '{video_name}' EXISTS in database. Used on: {used_date}, Clips created: {clips_created}"
        else:
            return f"Video '{video_name}' does NOT exist in database."

    except json.JSONDecodeError as e:
        return f"Error parsing database file: {str(e)}"
    except Exception as e:
        return f"Error checking video source: {str(e)}"



@tool
def update_used_videos(video_id: str, channel_name: str, title: str, published_at: str, description: str, video_file_name: str = None) -> None:
    """
    A tool that updates the record of used YouTube video IDs in a JSON file and
    mirrors the entry into Videosources_completed.json to prevent reprocessing.

    Args:
        video_id (str): The YouTube video ID to add to the used list.
        channel_name (str): The name of the channel.
        title (str): The title of the video.
        published_at (str): The publication date.
        description (str): The video description.
        video_file_name (str, optional): Local video file name (e.g., "My clip.mp4").
                                         If omitted, the title is used with an .mp4 extension.
    Returns:
        None
    """
    import json
    import time

    youtube_folder = os.path.join(REPO_ROOT, "Youtube", channel_name)
    os.makedirs(youtube_folder, exist_ok=True)

    json_file = os.path.join(youtube_folder, "used_videos.json")

    used_videos = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                used_videos = json.load(f)
        except Exception:
            used_videos = []


    if any(v.get("video_id") == video_id for v in used_videos):
        log(f"Video {video_id} already exists in records for {channel_name}.")
        return

    new_entry = {
        "video_id": video_id,
        "title": title,
        "published_at": published_at,
        "description": description,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    used_videos.append(new_entry)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(used_videos, f, indent=4, ensure_ascii=False)

    sources_file = os.path.join(REPO_ROOT, "work_queue_folder", "Videosources_completed.json")
    os.makedirs(os.path.dirname(sources_file), exist_ok=True)

    try:
        with open(sources_file, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except Exception:
        sources = {}

    safe_name = video_file_name if video_file_name else f"{title}.mp4"
    safe_name = os.path.basename(safe_name).strip()
    if not safe_name.lower().endswith(".mp4"):
        safe_name = f"{safe_name}.mp4"

    sources[safe_name] = {
        "used_date": time.strftime("%Y-%m-%d"),
        "clips_created": 0
    }

    with open(sources_file, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)

    log(f"Updated used_videos.json for {channel_name} with video {video_id} and tracked {safe_name} in Videosources_completed.json")



@tool
def youtube_Searcher(channel_name: str, search_query: str = None, max_results: int = 10, order: str = "relevance") -> dict:
    """
    A tool that retrieves information about videos from a specific YouTube channel.
    It can either list the latest uploads or search for videos matching a query within the channel.

    Args:
        channel_name (str): The name of the channel to look up (e.g., "Chris_williamson", "Jay_shetty", "Mel Robbins").
        search_query (str, optional): A keyword or topic to search for within the channel's videos. If None, fetches latest uploads (unless order is 'viewCount' or 'rating').
        max_results (int, optional): The maximum number of videos to return. Defaults to 10.
        order (str, optional): The order of the search results. Allowed values: 'date', 'rating', 'relevance', 'title', 'videoCount', 'viewCount'.
                               Defaults to 'relevance'. Use 'viewCount' to find the most popular videos.

    Returns:
        dict: A dictionary containing channel statistics and a list of videos (with titles, descriptions, etc.).
    """
    Youtube_channels = {
        "Chris_williamson": 'UCIaH-gZIVC432YRjNVvnyCA',
        "Jay_shetty": 'UCbk_QsfaFZG6PdQeCvaYXJQ',
        'Mel_Robbins': 'UCk2U-Oqn7RXf-ydPqfSxG5g',
        'Diary_ceo': 'UCGq-a57w-aPwyi3pW7XLiHw',
    }


    if channel_name not in Youtube_channels:
        return {"error": f"Channel '{channel_name}' not found in the available list. Available channels: {list(Youtube_channels.keys())}"}

    channel_id = Youtube_channels[channel_name]
    Api_key = os.getenv("YOUTUBE_API_KEY")
    if not Api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in environment variables.")

    youtube = build("youtube", "v3", developerKey=Api_key)

    channel_response = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        id=channel_id
    ).execute()

    if not channel_response.get("items"):
        return {"error": f"No details found for channel ID {channel_id}"}

    channel_info = channel_response["items"][0]
    uploads_playlist_id = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

    stats = {
        "title": channel_info["snippet"]["title"],
        "description": channel_info["snippet"]["description"],
        "subscriberCount": channel_info["statistics"]["subscriberCount"],
        "videoCount": channel_info["statistics"]["videoCount"],
        "viewCount": channel_info["statistics"]["viewCount"]
    }

    videos = []


    use_search_api = (search_query is not None) or (order in ["viewCount", "rating", "title", "videoCount"])

    if use_search_api:
        # Search within the channel
        search_params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": max_results,
            "order": order,
            "type": "video"
        }
        if search_query:
            search_params["q"] = search_query

        search_response = youtube.search().list(**search_params).execute()

        for item in search_response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": item["id"]["videoId"],
                "publishedAt": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"]
            })
    else:

        playlist_response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        ).execute()

        for item in playlist_response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "videoId": item["contentDetails"]["videoId"],
                "publishedAt": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"]
            })

    return {
        "channel_statistics": stats,
        "videos": videos
    }





@tool
def youtube_downloader(VideoId: str) -> str:
    """
    A tool that downloads a video based on a YouTube Video ID.

    Args:
        VideoId (str): The ID of the YouTube video to download.

    Returns:
        str: The absolute path to the downloaded video file, or None if the download fails.
    """
    download_dir = os.path.join(REPO_ROOT, "Video_clips", "Youtube_Upload_folder")
    os.makedirs(download_dir, exist_ok=True)

    yt_dlp_opts = {
        'quiet': False,
        'nocheckcertificate': True,
        'format': 'best',
        'debuge': True,
        'cookiefile': r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\youtube.com_cookies.txt",
        'cookiesfrombrowser': True,
        'outtmpl': os.path.join(download_dir, "%(title)s.%(ext)s"),
        'extractor_args': {
            'youtube': {
                'player_client': ['default', 'mweb'],
            }
        }
    }

    with yt_dlp.YoutubeDL(yt_dlp_opts) as ytdlp:
        try:
            video_url = f"https://www.youtube.com/watch?v={VideoId}"
            info_dict = ytdlp.extract_info(video_url, download=True)
            video_title = info_dict.get('title', "video")
            sanitized_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
            output_path = ytdlp.prepare_filename(info_dict)
            new_output_path = os.path.join(os.path.dirname(output_path), f"{sanitized_title}.mp4")

            if output_path != new_output_path:
                try:
                    os.rename(output_path, new_output_path)
                except FileNotFoundError:
                    # Some extractors return the final name already; keep the prepared filename.
                    new_output_path = output_path

            log(f"Downloaded video to: {new_output_path}")
            return new_output_path
        except Exception as e:
            log(f"DownloadError: {str(e)}")
            return None






DEFAULT_CHANNELS = ["Chris_williamson", "Jay_shetty"]
SYSTEM_PROMPT = SystemMessage(
	content=(
		"You are an inteligent and dedicated agent that can accomplish any goals you are given."
		"In order to accomplish your goals, you must apply chain of thought reasoning by thinking step by step and come up with a plan that solves the users query and execute the appropriate tools available for you."
  		""
	)
)
class AgentState(AgentState):
	messages: Annotated[List[BaseMessage],add]
	channels: List[str]
	downloaded_videos:  List[Dict[str,str]]

llm = ChatOpenAI(model="gpt-5", reasoning_effort="medium",max_tokens=16000)

agent = create_agent(
	model=llm,
 	tools= [youtube_Searcher, Check_Already_used_videos, check_video_source_exists, youtube_downloader],
  	state_schema=AgentState,
  	system_prompt=SYSTEM_PROMPT

)

agent.invoke()
