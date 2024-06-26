from huggingface_hub import HfApi, ModelFilter
import pandas as pd

def list_text_generation_models():
    api = HfApi()
    
    # You can filter by task and other criteria using ModelFilter
    filter = ModelFilter(task="text-generation")
    
    models = api.list_models(filter=filter, sort="downloads", direction=-1)
    
    return models

if __name__ == "__main__":
    models = list_text_generation_models()
    model_ids = [model.modelId for model in models]
    
    # Create a DataFrame with one column containing the model IDs
    df = pd.DataFrame(model_ids, columns=['Model ID'])
    
    # Write the DataFrame to a CSV file
    df.to_csv('data/text_generation_models.csv', index=False)
    
    print("Model IDs have been written to text_generation_models.csv")