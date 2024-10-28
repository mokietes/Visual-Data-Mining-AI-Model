from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Load the original dataset
dataset = load_dataset("miketes/filtered-non-english-wave-ui-25k")

# Define a filtering function to remove English rows
def filter_non_english(example):
    platform = example['platform']
    return platform is not None and 'mobile' in platform


# Filter the dataset using the defined function
filtered_dataset = dataset['train'].filter(filter_non_english)

# Print out some details to confirm it was filtered correctly
print("Filtered dataset preview:")
print(filtered_dataset)
print("Number of rows in the filtered dataset:", len(filtered_dataset))
print(dataset['train'].shuffle(seed=42).select(range(5)))

# Push the filtered dataset to Hugging Face Hub
# Log in with your Hugging Face API token in the terminal first
# huggingface-cli login

# Specify your username and repo name
username = "miketes"  # replace with your username or org name
repo_name = "mobile-filtered-english-wave-ui-25k"  # replace with your desired repo name

# Initialize the Hugging Face API and create the repository if needed
api = HfApi()
api.create_repo(repo_id=f"{username}/{repo_name}", repo_type="dataset", exist_ok=True)

# Push the filtered dataset to the repository
filtered_dataset.push_to_hub(f"{username}/{repo_name}")
