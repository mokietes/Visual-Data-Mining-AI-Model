import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from PIL import Image
from datasets import load_dataset
from huggingface_hub import login
import requests
from io import BytesIO

# Login to Hugging Face account
login(token='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')

# Model ID
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Dataset class for button detection
class ButtonDetectionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image']).convert('RGB')
        
        # Define the messages to send for processing the image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Find and describe the location of buttons in this image."}
                ]
            }
        ]
        
        # Define the target text format
        target_text = f"The button is located at coordinates x={item['button_x']}, y={item['button_y']}"
        
        # Process the image and text input
        inputs = processor(
            images=image,
            text=processor.apply_chat_template(messages, add_generation_prompt=True),
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        target_inputs = processor(
            text=target_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Flatten input tensors
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Assign labels to input for training
        inputs["labels"] = target_inputs.input_ids.squeeze(0)
        return inputs

# Main function to load the dataset and predict button location
def main():
    # Load dataset
    dataset = load_dataset("agentsea/wave-ui-25k")
    first_item = dataset['train'][0]
    image_data = first_item['image']
    
    # Print the path or URL of the first image in the dataset
    print("First image path or URL:", image_data)
    
    # Handle the image loading
    if isinstance(image_data, Image.Image):
        image = image_data.convert('RGB')
    elif isinstance(image_data, str):
        if image_data.startswith("http"):
            response = requests.get(image_data)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:  
            image = Image.open(image_data).convert('RGB')
    else:
        raise ValueError(f"Unexpected type for image data: {type(image_data)}")
    
    # Image loaded successfully
    print("Image loaded successfully")
    
    # Predict button location
    prediction = predict_button_location(image)
    print("Predicted Button Location:", prediction)

# Function to predict button location in the given image
def predict_button_location(image):
    # Define messages for the model input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Find and describe the location of buttons in this image."}
            ]
        }
    ]
    
    # Process the image and text input
    inputs = processor(
        images=image,
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        return_tensors="pt",
    )
    
    # Ensure only valid inputs are passed, and input_ids are correctly used
    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        raise ValueError("No input_ids found in the processed inputs.")

    # Generate prediction from the model
    outputs = model.generate(
        input_ids=input_ids,  # Ensure input_ids is passed to the model
        max_new_tokens=100,
        do_sample=False
    )
    
    # Decode the output and return the result
    return processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    main()
