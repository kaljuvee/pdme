from langchain.llms import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import openai
from google.generativeai import GenerativeModel
import anthropic
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

# New function for direct OpenAI API call
def get_openai_log_probs_direct(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            logprobs=True
        )
        return response.choices[0].logprobs
    except Exception as e:
        print(f"Error getting log probabilities from OpenAI: {e}")
        return None

# New function for direct Gemini API call
def get_gemini_response_direct(prompt):
    try:
        model = GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return None

# New function for direct Claude API call
def get_claude_response_direct(prompt):
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error getting response from Claude: {e}")
        return None

def main():
    load_dotenv()
    
    # Set up API keys
    openai.api_key = os.getenv("OPENAI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    # LangChain OpenAI
    print("Testing with LangChain OpenAI ChatGPT model:")
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    log_probs = get_log_probs_langchain(openai_llm, ("human", "How are you today?"))
    print(log_probs[:5] if log_probs else "No log probabilities available")
    print()

    # LangChain Gemini
    print("Testing with LangChain Google Generative AI model:")
    print("GOOGLE_API_KEY: ", os.environ["GOOGLE_API_KEY"])
    google_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    log_probs = get_log_probs_langchain(google_llm, ("human", "How are you today?"))
    print(log_probs[:5] if log_probs else "No log probabilities available")
    print()

    # LangChain Claude
    print("Testing with LangChain Claude model:")
    print("ANTHROPIC_API_KEY: ", os.environ["ANTHROPIC_API_KEY"])
    claude_llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    log_probs = get_log_probs_langchain(claude_llm, ("human", "How are you today?"))
    print(log_probs[:5] if log_probs else "No log probabilities available")
    print()

    # Direct OpenAI API
    #print("Testing with direct OpenAI API call:")
    #openai_log_probs = get_openai_log_probs_direct("How are you today?")
    #print(openai_log_probs[:5] if openai_log_probs else "No log probabilities available")
    #print()

    # Direct Gemini API
    #print('Checking Gemini key: ', os.environ["GOOGLE_API_KEY"])
    #print("Testing with direct Gemini API call:")
    #gemini_response = get_gemini_response_direct("How are you today?")
    #print(gemini_response if gemini_response else "No response available")
    #print()

    # Direct Claude API
    print('Checking Claude key: ', os.environ["ANTHROPIC_API_KEY"])
    print("Testing with direct Claude API call:")
    claude_response = get_claude_response_direct("How are you today?")
    print(claude_response if claude_response else "No response available")

if __name__ == "__main__":
    main()