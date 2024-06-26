import openai
import json
import argparse
import os
from dotenv import load_dotenv

class PDMEvaluator:
    def __init__(self, eval_model_api_key, model_1_api_key, model_2_api_key):
        self.eval_model_api_key = eval_model_api_key
        self.model_1_api_key = model_1_api_key
        self.model_2_api_key = model_2_api_key
    
    def generate_prompt(self, api_key, prompt, max_tokens=1000):
        openai.api_key = api_key
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose another engine if preferred
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=1,
            echo=True
        )
        return response.choices[0].text.strip(), response.choices[0].logprobs

    def generate_bootstrap_prompt(self, seed_1, seed_2, seed_3, seed_4):
        return f"Write a two sentence synopsis about {seed_1}, with the theme {seed_2}, and the story should somehow include {seed_3} and {seed_4}."

    def generate_question_prompt(self, bootstrap_prompt):
        question_prompt, _ = self.generate_prompt(self.eval_model_api_key, bootstrap_prompt)
        return question_prompt

    def get_model_responses(self, question_prompt):
        response_1, logprobs_1 = self.generate_prompt(self.model_1_api_key, question_prompt)
        response_2, logprobs_2 = self.generate_prompt(self.model_2_api_key, question_prompt)
        return (response_1, logprobs_1), (response_2, logprobs_2)
    
    def compare_logprobs(self, response1, response2):
        def get_response_logprob(logprobs, response):
            tokens = response.split()
            token_logprobs = logprobs['token_logprobs'][-len(tokens):]
            return sum(token_logprobs)

        response1_logprob = get_response_logprob(response1[1], response1[0])
        response2_logprob = get_response_logprob(response2[1], response2[0])

        response1_reversed_logprob = get_response_logprob(response1[1], response2[0])
        response2_reversed_logprob = get_response_logprob(response2[1], response1[0])

        prob1_better = (response1_logprob + response2_reversed_logprob) / 2
        prob2_better = (response2_logprob + response1_reversed_logprob) / 2

        if prob1_better > prob2_better:
            return [{'model': '1', 'rank': 1}, {'model': '2', 'rank': 2}]
        else:
            return [{'model': '1', 'rank': 2}, {'model': '2', 'rank': 1}]

    def compare_responses(self, question, response1, response2):
        vs_prompt = f"""
        <prefix><user_start>I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

        Here is the prompt:
        {{
            "instruction": "{question}",
        }}

        Here are the outputs of the models:
        [
            {{
                "model": "1",
                "answer": "{response1[0]}"
            }},
            {{
                "model": "2",
                "answer": "{response2[0]}"
            }}
        ]

        Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
        [
            {{'model': <model-name>, 'rank': <model-rank>}},
            {{'model': <model-name>, 'rank': <model-rank>}}
        ]

        Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
        <assistant_start>[
            {{'model': '"""
        return self.compare_logprobs(response1, response2)

    def evaluate(self, seed_1, seed_2, seed_3, seed_4):
        bootstrap_prompt = self.generate_bootstrap_prompt(seed_1, seed_2, seed_3, seed_4)
        question_prompt = self.generate_question_prompt(bootstrap_prompt)
        response1, response2 = self.get_model_responses(question_prompt)
        comparison = self.compare_responses(question_prompt, response1, response2)
        return comparison

def load_api_key(model_name):
    env_var_name = model_name.upper().replace("-", "_").replace("/", "_") + "_API_KEY"
    return os.getenv(env_var_name)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Prompt-Driven Model Evaluation")
    parser.add_argument('--eval_model', type=str, required=True, help='Name of the evaluation model')
    parser.add_argument('--model_1', type=str, required=True, help='Name of the first model to evaluate')
    parser.add_argument('--model_2', type=str, required=True, help='Name of the second model to evaluate')
    parser.add_argument('--seed_1', type=str, default="an old Englishman", help='Seed 1 for the bootstrap prompt')
    parser.add_argument('--seed_2', type=str, default="finding happiness", help='Seed 2 for the bootstrap prompt')
    parser.add_argument('--seed_3', type=str, default="rain", help='Seed 3 for the bootstrap prompt')
    parser.add_argument('--seed_4', type=str, default="old cars", help='Seed 4 for the bootstrap prompt')
    
    args = parser.parse_args()

    eval_model_api_key = load_api_key(args.eval_model)
    model_1_api_key = load_api_key(args.model_1)
    model_2_api_key = load_api_key(args.model_2)

    if not eval_model_api_key or not model_1_api_key or not model_2_api_key:
        raise ValueError("One or more API keys are missing. Please check your .env file.")

    pdme = PDMEvaluator(eval_model_api_key, model_1_api_key, model_2_api_key)
    result = pdme.evaluate(args.seed_1, args.seed_2, args.seed_3, args.seed_4)
    print(result)

if __name__ == "__main__":
    main()
