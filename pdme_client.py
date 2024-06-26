import argparse
import os
from dotenv import load_dotenv
from pdme_evaluator import PDMEvaluator
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

def load_model(model_name):
    if 'openai' in model_name.lower():
        openai_api_key = os.getenv('OPENAI_API_KEY')
        return OpenAI(temperature=0.9, api_key=openai_api_key)  # Adjust parameters as needed
    elif 'google' in model_name.lower():
        google_api_key = os.getenv('GOOGLE_API_KEY')
        return ChatGoogleGenerativeAI(model_name=model_name, api_key=google_api_key)
    else:
        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        llm  = HuggingFaceHub(repo_id = model_name,
                       model_kwargs={"temperature": 0, "max_length":200},
                       huggingfacehub_api_token=huggingface_api_key)
        return llm

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Prompt-Driven Model Evaluation")
    parser.add_argument('--eval_model', type=str, required=True, help='Name of the evaluation model')
    parser.add_argument('--test_model', type=str, required=True, help='Name of the test model')
    parser.add_argument('--seed_1', type=str, default="an old Englishman", help='Seed 1 for the bootstrap prompt')
    parser.add_argument('--seed_2', type=str, default="finding happiness", help='Seed 2 for the bootstrap prompt')
    parser.add_argument('--seed_3', type=str, default="rain", help='Seed 3 for the bootstrap prompt')
    parser.add_argument('--seed_4', type=str, default="old cars", help='Seed 4 for the bootstrap prompt')
    
    args = parser.parse_args()

    eval_model = load_model(args.eval_model)
    test_model = load_model(args.test_model)

    pdme = PDMEvaluator(eval_model, test_model)
    result = pdme.evaluate(args.seed_1, args.seed_2, args.seed_3, args.seed_4)
    print(result)

if __name__ == "__main__":
    main()
