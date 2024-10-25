# Step 1: Import necessary libraries
import os
from datasets import load_dataset, Dataset
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Step 2: Load the dataset from Hugging Face
dataset = load_dataset("agentsea/wave-ui-25k")

# Step 3: Define the key where the text is stored (e.g., 'text')
text_key = 'OCR'  # Replace with the actual key that holds the text in the dataset

# Step 4: Function to check if the text is in English
def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If language detection fails (e.g., text is too short), exclude the sample
        return False

# Step 5: Filter out non-English samples
def filter_english_samples(dataset_split):
    english_samples = [sample for sample in dataset_split if is_english(sample[text_key])]
    return english_samples

# Apply the filtering to the 'train' split of the dataset (you can also apply to 'test' or other splits)
english_samples = filter_english_samples(dataset['train'])

# Step 6: Convert the list of samples to a Hugging Face Dataset object
english_dataset = Dataset.from_list(english_samples)

# Step 7: Save the dataset in multiple formats (JSON, CSV, Hugging Face format)

# Create a folder to save the files if it doesn't exist
save_folder = "filtered_datasets"
os.makedirs(save_folder, exist_ok=True)

# Save as JSON
json_path = os.path.join(save_folder, "filtered_english_dataset.json")
english_dataset.to_json(json_path)
print(f"Dataset saved as JSON at: {json_path}")

# Save as CSV
csv_path = os.path.join(save_folder, "filtered_english_dataset.csv")
english_dataset.to_csv(csv_path)
print(f"Dataset saved as CSV at: {csv_path}")

# Save in Hugging Face dataset format (Arrow format)
arrow_path = os.path.join(save_folder, "filtered_english_dataset.arrow")
english_dataset.save_to_disk(arrow_path)
print(f"Dataset saved in Hugging Face format at: {arrow_path}")

