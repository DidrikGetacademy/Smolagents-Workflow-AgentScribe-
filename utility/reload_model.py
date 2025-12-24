from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
from utility.clean_memory import clean_get_gpu_memory
import  utility.Global_state
from smolagents import LiteLLMModel
from utility.log import log



###hva faen tenker jeg her..
class OpenAi_model():
    def __init__(self):
        self.model_id = model_id

  




def Reload_and_change_model(model_name, message):
    from App import Global_model

    if Global_model:
        del Global_model
        clean_get_gpu_memory(threshold=0.6)

    log(message)

    if model_name == "gpt-4o":
        utility.Global_state.set_current_global_model("gpt-4o")
        Global_model = LiteLLMModel(model_id="gpt-4o", api_key=OPENAI_APIKEY, temperature=0.0, max_tokens=16000)
        return Global_model

    elif model_name == "gpt-4-finetuned":
        utility.Global_state.set_current_global_model("gpt-4-finetuned")
        Global_model = LiteLLMModel(
            model_id="ft:gpt-4.1-2025-04-14:personal-learnreflects:motivational-short-extractor:CHeRJbsB",
            api_key=OPENAI_APIKEY,
            max_tokens=16384
        )
        return Global_model

    elif model_name == "gpt-5-medium":
        utility.Global_state.set_current_global_model("gpt-5")
        Global_model = LiteLLMModel(
            model_id="gpt-5",
            reasoning_effort="medium",
            api_key=OPENAI_APIKEY,
            max_tokens=20000
        )
        return Global_model
    elif model_name == "gpt-5-high":
        utility.Global_state.set_current_global_model("gpt-5-high")
        Global_model = LiteLLMModel(
            model_id="gpt-5",
            reasoning_effort="high",
            api_key=OPENAI_APIKEY,
            max_tokens=20000
        )
        return Global_model
    elif model_name == "gpt-5-minimal":
        utility.Global_state.set_current_global_model("gpt-5-minimal")
        Global_model = LiteLLMModel(
            model_id="gpt-5",
            reasoning_effort="minimal",
            api_key=OPENAI_APIKEY,
            max_tokens=16000
        )
        return Global_model
    elif model_name == "gpt-5.1":
        utility.Global_state.set_current_global_model("gpt-5.1")
        Global_model = LiteLLMModel(
            model_id="gpt-5.1",
            reasoning_effort="medium",
            api_key=OPENAI_APIKEY,
            max_tokens=16000
        )
        return Global_model
    elif model_name == "gpt-5.2":
        utility.Global_state.set_current_global_model("gpt-5.2")
        Global_model = LiteLLMModel(
            model_id="gpt-5.2",
            reasoning_effort="xhigh",
            api_key=OPENAI_APIKEY,
            max_tokens=16000
        )
        return Global_model
    elif model_name == "gpt-5-mini":
        utility.Global_state.set_current_global_model("gpt-5.2")
        Global_model = LiteLLMModel(
            model_id="gpt-5-mini",
            reasoning_effort="high",
            api_key=OPENAI_APIKEY,
            max_tokens=16000
        )
        return Global_model


    else:
        log(f"Unknown model name: {model_name}. Available models: gpt-4o, gpt-4-finetuned, gpt-5")
        raise ValueError(f"Unsupported model name: {model_name}")
