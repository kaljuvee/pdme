import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from pdme_evaluator import PDMEvaluator
from langchain_openai import OpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import HfApi

def get_model_id(model_name):
    return model_name.split('/')[-1]

def load_model(model_name):
    model_id = model_name
    try:        
        if 'openai' in model_name.lower():
            model_id = get_model_id(model_name)
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key not found in environment variables.")
            return OpenAI(model=model_id, temperature=0.2, api_key=openai_api_key).bind(logprobs=True)
        elif 'google' in model_name.lower():
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key:
                raise ValueError("Google API key not found in environment variables.")
            model_id = get_model_id(model_name)
            return ChatGoogleGenerativeAI(model_name=model_id, api_key=google_api_key)
        else:
            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not huggingface_api_key:
                raise ValueError("HuggingFace API key not found in environment variables.")
            llm = HuggingFaceHub(repo_id=model_id,
                                 model_kwargs={"temperature": 0.2, "max_length": 200},
                                 huggingfacehub_api_token=huggingface_api_key)
            return llm
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

def evaluate_models(eval_model_name, test_models, seeds):
    eval_model = load_model(eval_model_name)
    if eval_model is None:
        print("Failed to load the evaluation model. Exiting.")
        return

    for test_model_name in test_models:
        test_model = load_model(test_model_name)
        if test_model is None:
            print(f"Failed to load the test model '{test_model_name}'. Skipping.")
            continue
        
        pdme = PDMEvaluator(eval_model, test_model)
        result = pdme.evaluate(*seeds)
        print(f"Results for test model '{test_model_name}': {result}")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Prompt-Driven Model Evaluation")
    parser.add_argument('--eval_model', type=str, required=True, help='Name of the evaluation model')
    parser.add_argument('--test_model', type=str, help='Name of the test model')
    parser.add_argument('--test_model_file', type=str, help='CSV file containing test model IDs')
    parser.add_argument('--seed_1', type=str, default="an old Englishman", help='Seed 1 for the bootstrap prompt')
    parser.add_argument('--seed_2', type=str, default="finding happiness", help='Seed 2 for the bootstrap prompt')
    parser.add_argument('--seed_3', type=str, default="rain", help='Seed 3 for the bootstrap prompt')
    parser.add_argument('--seed_4', type=str, default="old cars", help='Seed 4 for the bootstrap prompt')
    
    args = parser.parse_args()

    test_models = []
    if args.test_model:
        test_models.append(args.test_model)
    if args.test_model_file:
        try:
            df = pd.read_csv(args.test_model_file)
            test_models.extend(df['model_id'].tolist())
        except Exception as e:
            print(f"Error reading test model file: {e}")
            return

    if not test_models:
        print("No test models provided. Exiting.")
        return

    seeds = [args.seed_1, args.seed_2, args.seed_3, args.seed_4]
    evaluate_models(args.eval_model, test_models, seeds)

if __name__ == "__main__":
    main()
