import os
import torch
import gc
from unsloth import FastLanguageModel
gc.collect()
torch.cuda.empty_cache()



# -----------------------------
# Paths
# -----------------------------
model_path = r""
train_file = os.path.join(model_path, "train.jsonl")
test_file = os.path.join(model_path, "test.jsonl")
output_dir = os.path.join(model_path, "")
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# Model and Tokenizer initalization/loading
# -----------------------------
max_seq_length = 2048
dtype=None
Load_in_4bit=True,
fourbit_models = [
    "unsloth/Qwen2-0.5b-bnb-4bit"
]

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=Load_in_4bit,
    )

# -----------------------------
#  PEFT (Parameter-Efficient Fine-Tuning)
# -----------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #It defines the dimension of the low-rank matrices used to approximate weight updates, controlling the parameter efficiency and capacity.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], #Specifies the names of the modules (layers) within the model to which the LoRA adaptation will be applied.
    lora_alpha=16, #lora_alpha is a scaling factor applied to the low-rank weights during the forward pass.It scales the updates to control their magnitude, often improving training stability.Here, a value of 16 means the LoRA updates are scaled by 16.
    lora_dropout=0,#no dropout is applied, so the LoRA layers always contribute fully during training.
    bias="none", #"none" means biases are not modified or trained during PEFT.
    use_gradient_checkpointing="unsloth", #Enables gradient checkpointing to save memory during training by recomputing parts of the graph on the backward pass.
    random_state=3407, #Sets the random seed or random state for reproducibility. Ensures that initialization and any randomness during PEFT application are deterministic.
    use_rslora=False,
    loftq_config=None, #Configuration for "loftq", which might be an additional method or module related to quantization or efficient tuning. None means no special loftq configuration is used. optional advanced feature.
)


# -----------------------------
# the Alpaca Prompt
# -----------------------------
alpaca_prompt  = """
                Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ### Instruction:
                {}
                ### Input:
                {}
                ### Response:
                {}"""
EOS_TOKEN = tokenizer.eos_token 
def formatting_prompts_func(exsamples):
    instructions = exsamples["instruction"]
    inputs = exsamples["input"]
    outputs = exsamples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text= alpaca_prompt.format(instruction,input,output) + EOS_TOKEN
        text.append(text)
    return {"text": texts,}



# -----------------------------
# Load and Preprocess the Dataset
# -----------------------------




#https://medium.com/@danushidk507/qwen2-finetuning-qwen2-f89c5c9d15da

