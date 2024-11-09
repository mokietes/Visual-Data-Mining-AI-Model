
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from PIL import Image
import json
import os
from datasets import load_dataset

from huggingface_hub import login
login(token='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')


class ButtonDetectionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image']).convert('RGB')
        
        # Create a prompt template for button detection
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Find and describe the location of buttons in this image."}
                ]
            }
        ]
        
        # Create target text based on button annotations
        target_text = f"The button is located at coordinates x={item['button_x']}, y={item['button_y']}"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Process target
        target_inputs = self.processor(
            text=target_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        inputs["labels"] = target_inputs.input_ids.squeeze(0)
        
        return inputs

def main():
    # Load the model and processor
    model_id = "/kaggle/input/llama-3.2-vision/transformers/11b-vision-instruct/1"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Load the Wave UI dataset
    dataset = load_dataset("agentsea/wave-ui-25k")
    
    # Create train and validation datasets
    train_dataset = ButtonDetectionDataset(dataset['train'], processor)
    val_dataset = ButtonDetectionDataset(dataset['validation'], processor)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./button-detection-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        gradient_accumulation_steps=4,
        fp16=True,  # Enable mixed precision training
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("./button-detection-model-final")

def predict_button_location(model, processor, image_path):
    """
    Function to predict button location in a new image
    """
    image = Image.open(image_path).convert('RGB')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Find and describe the location of buttons in this image."}
            ]
        }
    ]
    
    inputs = processor(
        images=image,
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        return_tensors="pt",
    )
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )
    
    return processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
    