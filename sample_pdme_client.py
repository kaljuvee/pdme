import argparse
import os
from dotenv import load_dotenv
from pdme_evaluator import PDME
from langchain_openai import OpenAI
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

def get_model_id(model_name):
    return model_name.split('/')[-1]

def validate_model_id(model_name):
    model_id = get_model_id(model_name)
    if model_name.lower().startswith('openai/'):
        return True
    else:
        url = f"https://huggingface.co/api/models/{model_id}"
        response = requests.head(url)
        if response.status_code == 200:
            return True
        else:
            print(f"Model ID '{model_id}' validation failed with status code {response.status_code}")
            return False

def load_model(model_name):
    model_id = model_name
    try:        
        if model_name.lower().startswith('openai/'):
            model_id = get_model_id(model_name)
            print('Evaluator (OpenAI) model: ', model_id)
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OpenAI API key not found in environment variables.")
            return OpenAI(model=model_id, temperature=0.2, api_key=openai_api_key)
        else:
            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
            #print(f"Hugging Face API key:, ', {huggingface_api_key}")
            if not huggingface_api_key:
                raise ValueError("HuggingFace API key not found in environment variables.")
            #if not validate_model_id(model_name):
            #    raise ValueError(f"Hugging Face model ID '{model_id}' is invalid or inaccessible.")
            login(huggingface_api_key)
            print('Test HuggingFace model: ', model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            return model, tokenizer
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None, None

def evaluate_models(eval_model_name, test_models, seeds):
    eval_model = load_model(eval_model_name)
    if eval_model is None:
        print("Failed to load the evaluation model. Exiting.")
        return

    for test_model_name in test_models:
        test_model, tokenizer = load_model(test_model_name)
        if test_model is None:
            print(f"Failed to load the test model '{test_model_name}'. Skipping.")
            continue
        
        pdme = PDME(eval_model, (test_model, tokenizer))
        result = pdme.evaluate(*seeds)
        print(f"Results for test model '{test_model_name}': {result}")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Prompt-Driven Model Evaluation")
    parser.add_argument('--eval_model', type=str, required=True, help='Name of the evaluation model')
    parser.add_argument('--test_model', type=str, help='Name of the test model')
    parser.add_argument('--seed_1', type=str, default="an old Englishman", help='Seed 1 for the bootstrap prompt')
    parser.add_argument('--seed_2', type=str, default="finding happiness", help='Seed 2 for the bootstrap prompt')
    parser.add_argument('--seed_3', type=str, default="rain", help='Seed 3 for the bootstrap prompt')
    parser.add_argument('--seed_4', type=str, default="old cars", help='Seed 4 for the bootstrap prompt')
    
    args = parser.parse_args()

    test_models = []
    if args.test_model:
        test_models.append(args.test_model)

    if not test_models:
        print("No test models provided. Exiting.")
        return

    seeds = [args.seed_1, args.seed_2, args.seed_3, args.seed_4]
    evaluate_models(args.eval_model, test_models, seeds)

if __name__ == "__main__":
    main()
