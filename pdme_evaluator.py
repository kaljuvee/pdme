import json
from langchain.llms import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PDMEvaluator:
    def __init__(self, eval_model, test_model):
        try:
            self.eval_model = eval_model
            self.test_model = test_model
        except Exception as e:
            print(f"Error initializing PDMEvaluator: {e}")
    
    def generate_prompt(self, model, prompt, max_tokens=1000):
        try:
            response = model(prompt, max_tokens=max_tokens)
            if isinstance(response, str):  # This handles the HuggingFace model
                return response.strip(), None
            return response['choices'][0]['text'].strip(), response['choices'][0].get('logprobs')
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return None, None

    def generate_bootstrap_prompt(self, seed_1, seed_2, seed_3, seed_4):
        try:
            return f"Write a two sentence synopsis about {seed_1}, with the theme {seed_2}, and the story should somehow include {seed_3} and {seed_4}."
        except Exception as e:
            print(f"Error generating bootstrap prompt: {e}")
            return ""

    def generate_question_prompt(self, bootstrap_prompt):
        try:
            question_prompt, _ = self.generate_prompt(self.eval_model, bootstrap_prompt)
            return question_prompt
        except Exception as e:
            print(f"Error generating question prompt: {e}")
            return ""

    def get_model_response(self, question_prompt):
        try:
            response, logprobs = self.generate_prompt(self.test_model, question_prompt)
            return response, logprobs
        except Exception as e:
            print(f"Error getting model response: {e}")
            return None, None
    
    def compare_logprobs(self, response1, response2):
        try:
            def get_response_logprob(logprobs, response):
                tokens = response.split()
                token_logprobs = logprobs['token_logprobs'][-len(tokens):]
                return sum(token_logprobs)

            response1_logprob = get_response_logprob(response1[1], response1[0]) if response1[1] else 0
            response2_logprob = get_response_logprob(response2[1], response2[0]) if response2[1] else 0

            response1_reversed_logprob = get_response_logprob(response1[1], response2[0]) if response1[1] else 0
            response2_reversed_logprob = get_response_logprob(response2[1], response1[0]) if response2[1] else 0

            prob1_better = (response1_logprob + response2_reversed_logprob) / 2
            prob2_better = (response2_logprob + response1_reversed_logprob) / 2

            if prob1_better > prob2_better:
                return [{'model': 'evaluation', 'rank': 1}, {'model': 'test', 'rank': 2}]
            else:
                return [{'model': 'evaluation', 'rank': 2}, {'model': 'test', 'rank': 1}]
        except Exception as e:
            print(f"Error comparing logprobs: {e}")
            return [{'model': 'evaluation', 'rank': 'unknown'}, {'model': 'test', 'rank': 'unknown'}]

    def compare_responses(self, question, response1, response2):
        try:
            vs_prompt = f"""
            <prefix><user_start>I want you to create a leaderboard of different large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

            Here is the prompt:
            {{
                "instruction": "{question}",
            }}

            Here are the outputs of the models:
            [
                {{
                    "model": "evaluation",
                    "answer": "{response1[0]}"
                }},
                {{
                    "model": "test",
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
                {{'model': 'evaluation', 'rank': """
            return self.compare_logprobs(response1, response2)
        except Exception as e:
            print(f"Error comparing responses: {e}")
            return [{'model': 'evaluation', 'rank': 'unknown'}, {'model': 'test', 'rank': 'unknown'}]

    def evaluate(self, seed_1, seed_2, seed_3, seed_4):
        try:
            bootstrap_prompt = self.generate_bootstrap_prompt(seed_1, seed_2, seed_3, seed_4)
            question_prompt = self.generate_question_prompt(bootstrap_prompt)
            eval_response, eval_logprobs = self.generate_prompt(self.eval_model, question_prompt)
            test_response, test_logprobs = self.get_model_response(question_prompt)
            comparison = self.compare_responses(question_prompt, (eval_response, eval_logprobs), (test_response, test_logprobs))
            return comparison
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return []
