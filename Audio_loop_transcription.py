from Custom_Agent_Tools import SpeechToTextToolCUDA
import os
if __name__ == "__main__":
    tool = SpeechToTextToolCUDA()
    tool.setup()
    audio_folder = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\test_audio"
    count = 0
    for filename in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder,filename)
        print(f"file_path {file_path}")
        count += 1
        tool.forward({"audio": file_path, "text_path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\temp.txt",  "video_path": None})
