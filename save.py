# Step 1: Import the Hugging Face dataset library
from datasets import load_dataset

# Step 2: Load the dataset from Hugging Face
dataset = load_dataset("agentsea/wave-ui-25k")

# Step 3: Save the dataset to a local directory (Hugging Face Arrow format)
save_path = "wave-ui-25k-dataset"
dataset.save_to_disk(save_path)

print(f"Dataset saved at {save_path}")
