import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch

class PDME:
    def __init__(self, eval_model, test_model):
        """Initialize the PDME class with evaluation and test models."""
        try:
            self.eval_model = eval_model
            self.test_model, self.test_tokenizer = test_model
        except Exception as e:
            print(f"Error initializing PDME: {e}")

    def get_token_logprobs(self, logprobs, response, tokenizer):
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
        
    def get_text_and_log_probs_langchain(self, llm, prompt):
        """Get both generated text and log probabilities from LangChain model."""
        try:
            if hasattr(llm, 'bind') and callable(llm.bind):
                bound_llm = llm.bind(logprobs=True)
                response = bound_llm.invoke(prompt)
                generated_text = response.content
                log_probs = response.response_metadata.get("logprobs", {}).get("content", [])
                print("Type for log_probs for OpenAI / LangChain  model: ", type(log_probs))
                print("Top k log_probs for OpenAI / LangChain  model: ", log_probs[:5])
                return generated_text, log_probs
            else:
                response = llm(prompt)
                print("Log probabilities not available for this model.")
                return response, None
        except Exception as e:
            print(f"Error getting log probabilities: {e}")
            return None, None

    def generate_prompt(self, model, prompt, max_tokens=1000):
        """Generate a prompt using the given model."""
        try:
            if isinstance(model, PreTrainedModel):
                print("Generating prompt and log_probs for Hugging Face model")
                tokenizer = self.test_tokenizer
                
                inputs = tokenizer(prompt, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                
                generated_ids = model.generate(**inputs, max_length=max_tokens)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)                
                log_probs = outputs.logits.log_softmax(dim=-1)
                print("Type for log_probs for HuggingFace model: ", type(log_probs))
                return generated_text, log_probs
            else:
                print("Generating prompt for OpenAI / LangChain model")
                generated_text, logprobs = self.get_text_and_log_probs_langchain(model, prompt)
                
                return generated_text, logprobs
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return None, None

    def generate_bootstrap_prompt(self, seed_1, seed_2, seed_3, seed_4):
        """Generate a bootstrap prompt using the given seeds."""
        try:
            return f"Write a two sentence synopsis about {seed_1}, with the theme {seed_2}, and the story should somehow include {seed_3} and {seed_4}."
        except Exception as e:
            print(f"Error generating bootstrap prompt: {e}")
            return ""

    def generate_question_prompt(self, bootstrap_prompt):
        """Generate a question prompt using the bootstrap prompt."""
        try:
            question_prompt, _ = self.generate_prompt(self.eval_model, bootstrap_prompt)
            print(f"Generated question prompt: {question_prompt}")
            return question_prompt
        except Exception as e:
            print(f"Error generating question prompt: {e}")
            return ""

    def get_model_response(self, question_prompt):
        """Get the response from the test model for the given question prompt."""
        try:
            response, logprobs = self.generate_prompt(self.test_model, question_prompt)
            return response, logprobs
        except Exception as e:
            print(f"Error getting model response: {e}")
            return None, None

    def compare_logprobs(self, response1, response2):
        """Compare log probabilities of two responses."""
        try:
            response1_logprob = self.get_token_logprobs(response1[1], response1[0], self.test_tokenizer)
            response2_logprob = self.get_token_logprobs(response2[1], response2[0], self.test_tokenizer)

            response1_reversed_logprob = self.get_token_logprobs(response1[1], response2[0], self.test_tokenizer)
            response2_reversed_logprob = self.get_token_logprobs(response2[1], response1[0], self.test_tokenizer)

            print(f"Evaluation model log probability: {response1_logprob}")
            print(f"Test model log probability: {response2_logprob}")
            print(f"Evaluation model reversed log probability: {response1_reversed_logprob}")
            print(f"Test model reversed log probability: {response2_reversed_logprob}")

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
        """Compare responses of two models."""
        try:
            return self.compare_logprobs(response1, response2)
        except Exception as e:
            print(f"Error comparing responses: {e}")
            return [{'model': 'evaluation', 'rank': 'unknown'}, {'model': 'test', 'rank': 'unknown'}]

    def evaluate(self, seed_1, seed_2, seed_3, seed_4):
        """Evaluate the models using the given seeds."""
        try:
            bootstrap_prompt = self.generate_bootstrap_prompt(seed_1, seed_2, seed_3, seed_4)
            print(f"Bootstrap prompt: {bootstrap_prompt}")
            question_prompt = self.generate_question_prompt(bootstrap_prompt)
            eval_response, eval_logprobs = self.generate_prompt(self.eval_model, question_prompt)
            test_response, test_logprobs = self.get_model_response(question_prompt)
            comparison = self.compare_responses(question_prompt, (eval_response, eval_logprobs), (test_response, test_logprobs))
            return comparison
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return []
