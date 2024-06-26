# Prompt Driven Model Evaluation (PMDE)

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
 ```
python pdme_evaluator.py 
    --eval_model meta-llama/Meta-Llama-3-8B 
    --generation_model text-davinci-003 
    --model_1 deepseek-ai/DeepSeek-Coder-V2-Instruct 
    --model_2 meta-llama/Meta-Llama-3-8B-Instruct 
    --seed_1 "an old Englishman" 
    --seed_2 "finding happiness" 
    --seed_3 "rain" 
    --seed_4 "old cars"

 ```