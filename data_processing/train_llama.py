import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load your filtered dataset
dataset = load_dataset("miketes/Web-filtered-english-wave-ui-25k")

# Define input and target columns for text generation
def format_data(example):
    # Combine multiple fields into a single input string
    input_text = (
        f"Instruction: {example['instruction']}\n"
        f"Image: {example['image']}\n"
        f"BBox: {example['bbox']}\n"
        f"Resolution: {example['resolution']}\n"
        f"Name: {example['name']}\n"
        f"Description: {example['description']}\n"
        f"Type: {example['type']}\n"
        f"Purpose: {example['purpose']}\n"
    )
    target_text = example['expectation']
    return {"input": input_text, "target": target_text}

dataset = dataset.map(format_data)

# Load tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Ensure model is on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize
def tokenize_function(example):
    inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=256)
    targets = tokenizer(example["target"], truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up Trainer with GPU specifications
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=4,   # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,                      # Mixed-precision training for A100 efficiency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("llama-finetuned")
tokenizer.save_pretrained("llama-finetuned")
