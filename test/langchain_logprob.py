from langchain.llms import BaseLLM
from langchain_openai import ChatOpenAI
import time
import os
from dotenv import load_dotenv

# Original LangChain function
def get_log_probs_langchain(llm, prompt):
    try:
        # use introspection to check available functions
        if hasattr(llm, 'bind') and callable(llm.bind):
            bound_llm = llm.bind(logprobs=True)
            response = bound_llm.invoke(prompt)
            return response.response_metadata.get("logprobs", {}).get("content", [])
        else:
            # For models that don't support binding or logprobs
            response = llm(prompt)
            print("Log probabilities not available for this model.")
            return None
    except Exception as e:
        print(f"Error getting log probabilities: {e}")
        return None

def main():
    load_dotenv()
    
    # LangChain OpenAI
    print("Testing with LangChain OpenAI ChatGPT model:")
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    log_probs = get_log_probs_langchain(openai_llm, ("human", "How are you today?"))
    print(log_probs[:5] if log_probs else "No log probabilities available")
    print()


if __name__ == "__main__":
    main()