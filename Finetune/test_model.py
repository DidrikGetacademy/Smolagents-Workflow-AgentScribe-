import json
import torch
import yaml
from typing import Dict, List
from dataclasses import dataclass
from smolagents import CodeAgent, TransformersModel, FinalAnswerTool

@dataclass
class Chunk:
    text: str
    label: str  # "Keep_T1", "Keep_T2", "Discard"
    position: str
    delivery_cues: List[str]

# Expanded Test Data with Detailed Examples
TEST_CHUNKS = [
    Chunk(text="Success is rented, not owned! [LOUD] And the rent is due every morning at dawn. [CHEERS]",
          label="Keep_T1", position="00:05:23", delivery_cues=["LOUD", "CHEERS"]),
    Chunk(text="The obstacle is the way - but only if you bring a map. [PAUSE] What's your navigation plan?",
          label="Keep_T1", position="00:07:12", delivery_cues=["PAUSE"]),
    Chunk(text="Just keep going, you've got this! [PAUSE]",
          label="Keep_T2", position="00:12:41", delivery_cues=["PAUSE"]),
    Chunk(text="No risk? [WHISPER] No story! [LOUD]",
          label="Keep_T2", position="00:15:30", delivery_cues=["WHISPER", "LOUD"]),
    Chunk(text="What if I told you... [WHISPER] your comfort zone is a prison you built yourself? [SILENCE]",
          label="Keep_T2", position="00:27:15", delivery_cues=["WHISPER", "SILENCE"]),
    Chunk(text="Dreams without deadlines are just fantasies. [SLOW]",
          label="Keep_T1", position="00:33:22", delivery_cues=["SLOW"]),
    Chunk(text="Believe in yourself and work hard.",
          label="Discard", position="00:18:09", delivery_cues=[]),
    Chunk(text="Stay positive and keep smiling!",
          label="Discard", position="00:21:45", delivery_cues=[]),
    Chunk(text="This changes everything. [PAUSE]",
          label="Discard", position="00:44:10", delivery_cues=["PAUSE"]),
    Chunk(text="...and THAT'S how you do it! [CHEERS]",
          label="Keep_T2", position="01:02:55", delivery_cues=["CHEERS"]),
    Chunk(text="You. [PAUSE] Yes YOU. [LOUD] Are you watching or working?",
          label="Keep_T2", position="01:15:30", delivery_cues=["PAUSE", "LOUD"]),
    Chunk(text="Small steps [WHISPER] lead to big leaps [LOUD] over time.",
          label="Keep_T1", position="01:22:10", delivery_cues=["WHISPER", "LOUD"]),
    Chunk(text="Success is rented, not owned! [LOUD] And the rent is due every morning at dawn. [CHEERS]",
          label="Keep_T1", position="00:05:23", delivery_cues=["LOUD", "CHEERS"]),
    Chunk(text="Just keep going, you've got this! [PAUSE]",
          label="Keep_T2", position="00:12:41", delivery_cues=["PAUSE"]),
    Chunk(text="Believe in yourself and work hard.",
          label="Discard", position="00:18:09", delivery_cues=[]),
]

SYSTEM_PROMPT = """Act as a Motivational Clip Analyst. Analyze this chunk:

Text: {text}
Position: {position}
Delivery Cues: {cues}

Respond ONLY with JSON: {{ "verdict": "Keep_T1|Keep_T2|Discard", "reason": string, "confidence": 0-10 }}"""

class MotivationAnalyzer:
    def __init__(self, agent):
        self.agent = agent
        
    def format_prompt(self, chunk: Chunk) -> str:
        features = ", ".join(chunk.features) or "None"
        return SYSTEM_PROMPT.format(
            text=chunk.text,
            position=chunk.position,
            features=features
        )
    def analyze_chunk(self, chunk: Chunk) -> Dict:
        prompt = self.format_prompt(chunk)
        response = self.agent.run(task=prompt)
        try:
            if isinstance(response, str):
                # Handle extra text before JSON
                json_str = response[response.find("{"):response.rfind("}")+1]
                return json.loads(json_str)
            return response
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}

def run_evaluation(analyzer: MotivationAnalyzer) -> Dict:
    results = []
    
    for chunk in TEST_CHUNKS:
        analysis = analyzer.analyze_chunk(chunk)
        results.append({
            "text": chunk.text,
            "expected": chunk.label,
            "actual": analysis.get("verdict"),
            "confidence": analysis.get("confidence", 0),
            "match": chunk.label == analysis.get("verdict")
        })
    
    accuracy = sum(1 for r in results if r['match']) / len(results)
    return {
        "accuracy": f"{accuracy:.1%}",
        "details": results
    }

if __name__ == "__main__":
    model = TransformersModel(
        model_id=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-7B-Instruct",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    with open(r'C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Testing_Prompt.yaml', 'r', encoding='utf-8') as f:
        prompt_templates = yaml.safe_load(f)
    
    agent = CodeAgent(
        model=model,
        max_steps=3,
        prompt_templates=prompt_templates,
        tools=[FinalAnswerTool()]
    )
    
    analyzer = MotivationAnalyzer(agent)
    results = run_evaluation(analyzer)
    
    print(f"\nTest Accuracy: {results['accuracy']}")
    for idx, result in enumerate(results['details']):
        status = "✅" if result['match'] else "❌"
        print(f"\nChunk {idx+1}:")
        print(f"Text: {result['text'][:60]}...")
        print(f"Expected: {result['expected']} | Actual: {result['actual']}")
        print(f"Match: {status} | Confidence: {result['confidence']}")
