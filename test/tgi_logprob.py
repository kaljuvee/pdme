import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceApi
from dotenv import load_dotenv
import os
from huggingface_hub import login



def get_log_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_probs = outputs.logits.log_softmax(dim=-1)
    return log_probs

def main():
    load_dotenv()
    huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
    # print(f"Hugging Face API key:, ', {huggingface_api_key}")
    if huggingface_api_key:
        login(huggingface_api_key)
    else:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set")

    model_name = "EleutherAI/gpt-neo-125m"
    print('Model name: ', model_name)
    # google/gemma-2b, openai-community/gpt2, EleutherAI/gpt-neo-2.7B, distilbert/distilgpt2, meta-llama/Llama-2-7b-hf, EleutherAI/gpt-neo-125m, 
    # microsoft/Phi-3-medium-128k-instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Once upon a time"
    log_probs = get_log_probs(model, tokenizer, prompt)
    print(log_probs)

if __name__ == "__main__":
    
    main()
