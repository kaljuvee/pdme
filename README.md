# Prompt Driven Model Evaluation (PDME)

## Step 1: Set Up Virtual Environment and Install Dependencies

### Clone the Repo
  ```
  git clone https://github.com/opticonomy/pdme.git
  ```

### Create and Activate the Virtual Environment
- Set up a Python virtual environment and activate it (Linux):
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- Set up a Python virtual environment and activate it (Windows/VS Code / Bash):
  ```
  python -m venv venv
  source venv/Scripts/activate
  ```
  
- Install dependencies from the `requirements.txt` file:
  ```
  pip install -r requirements.txt
  ```

### Sample Usage
#### Single test model
 ```
python sample_pdme_client.py --eval_model openai/gpt-4o --test_model microsoft/Phi-3-mini-4k-instruct --seed_1 "an old Englishman" --seed_2 "finding happiness" --seed_3 "rain" --seed_4 "old cars"
python sample_pdme_client.py --eval_model openai/gpt-3.5-turbo-instruct --test_model microsoft/Phi-3-mini-4k-instruct --seed_1 "an old Englishman" --seed_2 "finding happiness" --seed_3 "rain" --seed_4 "old cars"
python sample_pdme_client.py --eval_model openai/gpt-4o --test_model meta-llama/Meta-Llama-Guard-2-8B --seed_1 "an old Englishman" --seed_2 "finding happiness" --seed_3 "rain" --seed_4 "old cars"
python sample_pdme_client.py --eval_model openai/gpt-3.5-turbo-instruct --test_model meta-llama/Meta-Llama-Guard-2-8B --seed_1 "an old Englishman" --seed_2 "finding happiness" --seed_3 "rain" --seed_4 "old cars"
 ```
#### Multiple test models from a file
```
python pdme_client.py --eval_model openai/gpt-4o --test_model_file data/hf_text_generation_models.csv --seed_1 "an old Englishman" --seed_2 "finding happiness" --seed_3 "rain" --seed_4 "old cars"
 ```
 ## Overview

The method uses a single text generation AI, referred to as eval model, to evaluate any other text generation AI on any topic, and the evaluation works like this:

1. We write a text prompt for what questions the eval model should generate, and provide seeds that are randomly picked to generate a question.
2. The question is sent to the AI model being tested, and it generates a response.
3. Likewise, the eval model also generates an answer to the same question.
4. The eval model then uses a text prompt we write, to compare the two answers and pick the winner. (This model does not necessarily have to be the same as the eval model, but it does simplify inference)

This method allows us to evaluate models for any topic, such as: storytelling, programming, finance, and QnA.

## Technical Description

See above for the installation and running instructions.

### Example Use Case

Let’s say you want to evaluate a model's ability to write stories, PDME should be possible to use in the following way:

1. **Bootstrap Prompt** - First generate a bootstrap prompt using random seeds, e.g.

(continue....)

### Resources
- [LangChain Providers](https://python.langchain.com/v0.2/docs/integrations/platforms/)
