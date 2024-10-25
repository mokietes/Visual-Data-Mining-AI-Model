from datasets import load_dataset, Dataset
import pandas as pd

# Load the original dataset
dataset = load_dataset("agentsea/wave-ui-25k")

# Convert the dataset to a pandas DataFrame
df = dataset['train'].to_pandas()

# Print the column names to verify
print("Column names:", df.columns)

# Inspect unique values in the language column
print("Unique language values:", df['language'].unique())

# Filter for rows that contain 'English' in the language column (case insensitive)
df_filtered = df[df['language'].str.contains('English', case=False, na=False)]

# Convert the filtered DataFrame back to a Hugging Face Dataset
dataset_filtered = Dataset.from_pandas(df_filtered)

# Save the filtered dataset locally
dataset_filtered.save_to_disk("filtered_dataset")

# Load the filtered dataset back to check if it saved correctly
loaded_filtered_dataset = Dataset.load_from_disk("filtered_dataset")

# Print out some details to confirm it was filtered correctly
print("Filtered dataset preview:")
print(loaded_filtered_dataset)
print("Number of rows in the filtered dataset:", len(loaded_filtered_dataset))
