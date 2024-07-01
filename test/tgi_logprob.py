import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceApi

# Function to get log probabilities
def get_log_probs(model, tokenizer, prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Forward pass through the model to get outputs
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
    
    # Get the log probabilities
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    
    return log_probs

# Main function
def main():
    model_name = "gpt2"  # You can replace this with any other model available on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Example prompt
    prompt = "Once upon a time"

    # Get the log probabilities
    log_probs = get_log_probs(model, tokenizer, prompt)

    # Print the log probabilities
    print(log_probs)

if __name__ == "__main__":
    main()
