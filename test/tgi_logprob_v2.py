import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

def get_log_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_probs = outputs.logits.log_softmax(dim=-1)
    return log_probs

def main():
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
    print(f"Hugging Face API key:, ', {huggingface_api_key}")
    if huggingface_api_key:
        login(huggingface_api_key)
    else:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set")

    model_name = "google/gemma-2b"
    # google/gemma-2b, gpt2, EleutherAI/gpt-neo-2.7B, distilGPT2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Once upon a time"
    log_probs = get_log_probs(model, tokenizer, prompt)
    print(log_probs)

if __name__ == "__main__":
    main()
