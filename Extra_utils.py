
from faster_whisper import WhisperModel
import torch 
import time
from log import log 
class SpeechToTextTool_verify():
    default_checkpoint = r"c:\Users\didri\Desktop\LLM-models\Audio-Models\faster-whisper-large-v3-turbo-int8float16"
    description = "Fast tool that transcribes audio into text using faster-whisper. It returns the path to the transcript file"
    name = "Transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, a URL, or a tensor.",
        },
    }
    output_type = "string"
    def setup(self):

        self.model = WhisperModel(
                model_size_or_path=self.default_checkpoint,
                device="cuda",
                compute_type="int8_float16"
                )
        

    def forward(self, inputs):
        audio_path = inputs["audio"]
        segments, info = self.model.transcribe(
            audio_path,
            #vad_filter=True,
            #vad_parameters={"min_silence_duration_ms": 0},
            word_timestamps=True  
        )

        print(f"[INFO] Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
        print(f"[INFO] Audio Duration: {info.duration:.2f} seconds")
        segments_list = []
        for segment in segments:
            log(f"segment appended: {segment}")
            segment_dict = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [  
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    }
                    for word in segment.words
                ] if segment.words else []
            }
            segments_list.append(segment_dict)

    
        if self.device == "cuda":
                torch.cuda.empty_cache()
                del self.model 
        log(f"[SpeechToTextTool_verify]segment dict: {segment_dict}")

        return segments_list

    def encode(self, audio):
        return {"audio": audio}

    def decode(self, outputs):
        return outputs
