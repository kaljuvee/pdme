from langchain.llms import BaseLLM
from langchain_openai import ChatOpenAI
import time
import os
from dotenv import load_dotenv

# Function to get both generated text and log probabilities from LangChain model
def get_text_and_log_probs_langchain(llm, prompt):
    try:
        # Use introspection to check available functions
        if hasattr(llm, 'bind') and callable(llm.bind):
            bound_llm = llm.bind(logprobs=True)
            response = bound_llm.invoke(prompt)
            generated_text = response.content
            log_probs = response.response_metadata.get("logprobs", {}).get("content", [])
            return generated_text, log_probs
        else:
            # For models that don't support binding or logprobs
            response = llm(prompt)
            print("Log probabilities not available for this model.")
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
    print('model:', model_id)
    prompt = 'Write a two sentence synopsis about an old Englishman, with the theme finding happiness, and the story should somehow include rain and old cars.'
    generated_text, log_probs = get_text_and_log_probs_langchain(openai_llm, prompt)
    print("Generated text:", generated_text)
    print("Log probabilities:", log_probs[:5] if log_probs else "No log probabilities available")

if __name__ == "__main__":
    main()
