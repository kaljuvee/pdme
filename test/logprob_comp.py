import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_token_logprobs(logprobs, response, tokenizer):
    """Calculate the sum of log probabilities for the given response tokens."""
    try:
        if logprobs is None:
            return 0
        if isinstance(logprobs, list):
            print(f"Logprobs as list: {logprobs[:10]}")
            tokens = tokenizer(response, return_tensors='pt')['input_ids'][0]
            token_logprobs = [logprobs[i]['logprob'] for i in range(min(10, len(logprobs)))]
            return sum(token_logprobs)
        else:
            print(f"Logprobs as tensor: {logprobs.size()}")
            tokens = tokenizer(response, return_tensors='pt')['input_ids'][0]
            logprobs = logprobs.squeeze(0)

            # Ensure the lengths match
            len_diff = len(tokens) - logprobs.size(0)
            if len_diff > 0:
                tokens = tokens[:logprobs.size(0)]
            elif len_diff < 0:
                logprobs = logprobs[:len(tokens), :]

            token_logprobs = logprobs[torch.arange(len(tokens)), tokens][:10]
            return token_logprobs.sum().item()
    except Exception as e:
        print(f"Error in get_token_logprobs: {e}")
        return 0

def get_text_and_log_probs_langchain(llm, prompt):
    """Get both generated text and log probabilities from LangChain model."""
    try:
        if hasattr(llm, 'bind') and callable(llm.bind):
            bound_llm = llm.bind(logprobs=True)
            response = bound_llm.invoke(prompt)
            generated_text = response.content
            log_probs = response.response_metadata.get("logprobs", {}).get("content", [])
            return generated_text, log_probs
        else:
            response = llm(prompt)
            return response, None
    except Exception as e:
        print(f"Error getting log probabilities: {e}")
        return None, None

def main():
    load_dotenv()
    model_id = "gpt-3.5-turbo-0125"

    # LangChain OpenAI
    print("Testing with LangChain OpenAI ChatGPT model:")
    openai_llm = ChatOpenAI(model=model_id)
    prompt = 'Write a two sentence synopsis about an old Englishman, with the theme finding happiness, and the story should somehow include rain and old cars.'
    generated_text, log_probs = get_text_and_log_probs_langchain(openai_llm, prompt)
    print("Generated text:", generated_text)
    print("Log probabilities:", log_probs[:5] if log_probs else "No log probabilities available")

    if log_probs:
        token_logprobs_sum = get_token_logprobs(log_probs, generated_text, AutoTokenizer.from_pretrained("gpt2"))
        print("Log probabilities (sum) for LangChain model:", token_logprobs_sum)

    # Hugging Face
    print("\nTesting with Hugging Face GPT-2 model:")
    test_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(test_model_name)
    model = AutoModelForCausalLM.from_pretrained(test_model_name)
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    generated_ids = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    log_probs = outputs.logits.log_softmax(dim=-1)

    token_logprobs_sum = get_token_logprobs(log_probs, generated_text, tokenizer)
    
    print("Generated text:", generated_text)
    print("Log probabilities (sum) for Hugging Face model:", token_logprobs_sum)

if __name__ == "__main__":
    main()
