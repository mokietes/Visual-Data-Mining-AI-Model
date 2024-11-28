import torch
import gc
from contextlib import contextmanager
from datasets import load_dataset
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import math
import numpy as np

class ModelMemoryManager:
   @contextmanager
   def load_model(self, model_class, model_name, processor_class=None, peft_model_path=None, **kwargs):
       try:
           model = model_class.from_pretrained(model_name, **kwargs)
           if peft_model_path:
               from peft import PeftModel
               model = PeftModel.from_pretrained(model, peft_model_path)
           processor = processor_class.from_pretrained(model_name) if processor_class else None
           yield model, processor
       finally:
           del model
           if processor:
               del processor
           torch.cuda.empty_cache()
           gc.collect()

def convert_to_pixels(bbox, image_size):
   x_res, y_res = image_size
   return [
       bbox[0] * x_res / 100,
       bbox[1] * y_res / 100,
       bbox[2] * x_res / 100,
       bbox[3] * y_res / 100
   ]

def calculate_distance(point1, point2):
   return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_batch(model, processor, dataset, start_index, batch_size, show_images=False):
   # Initialize error tracking
   total_error_p1 = 0
   total_error_p2 = 0
   errors_p1 = []  # List to store all errors for std calculation
   errors_p2 = []
   successful_predictions = 0
   failed_predictions = []
   
   for i in range(batch_size):
       index = start_index + i
       image = dataset[index]['images']
       prompt = dataset[index]['texts'][0]['user']
       
       # Process image directly without temp file
       img_for_model = image.convert("RGB")
       conversation = [
           {
               "role": "user",
               "content": [{"type": "image"}, {"type": "text", "text": prompt}],
           }
       ]
       prompt_text = processor.apply_chat_template(
           conversation, add_generation_prompt=True, tokenize=False
       )
       inputs = processor(img_for_model, prompt_text, return_tensors="pt").to(model.device)
       output = model.generate(
           **inputs,
           temperature=0.5,
           top_p=0.8,
           max_new_tokens=512,
       )
       pred = processor.decode(output[0])[len(prompt_text):]
       
       # Process boxes after model inference
       true_box = [float(x) for x in dataset[index]['texts'][0]['assistant'].strip('[]').split(',')]
       true_box_pixels = convert_to_pixels(true_box, image.size)
       
       try:
           pred_numbers = pred.split('|>')[1].split('<|')[0].strip()
           pred_box = [float(x) for x in pred_numbers.strip('[]').split(',')]
           pred_box_pixels = convert_to_pixels(pred_box, image.size)
           
           # Calculate distances for both points
           p1_distance = calculate_distance(
               [true_box_pixels[0], true_box_pixels[1]], 
               [pred_box_pixels[0], pred_box_pixels[1]]
           )
           p2_distance = calculate_distance(
               [true_box_pixels[2], true_box_pixels[3]], 
               [pred_box_pixels[2], pred_box_pixels[3]]
           )
           
           total_error_p1 += p1_distance
           total_error_p2 += p2_distance
           errors_p1.append(p1_distance)
           errors_p2.append(p2_distance)
           successful_predictions += 1
           
           if show_images:
               draw = ImageDraw.Draw(image)
               draw.rectangle(true_box_pixels, outline='green', width=3)
               draw.rectangle(pred_box_pixels, outline='red', width=3)
               plt.figure(figsize=(12,8))
               plt.imshow(image)
               plt.axis('off')
               plt.show()
               plt.close()
               
               # Print metrics after each image
               print(f"\nMetrics for image {index}:")
               print(f"Point 1 (x1,y1) error: {p1_distance:.2f} pixels")
               print(f"Point 2 (x2,y2) error: {p2_distance:.2f} pixels")
               print(f"Average error: {(p1_distance + p2_distance)/2:.2f} pixels")
           
       except Exception as e:
           print(f"Error parsing prediction at index {index}: {e}")
           print(f"Raw prediction: {pred}")
           failed_predictions.append(index)
   
   # Print final statistics
   if successful_predictions > 0:
       avg_error_p1 = total_error_p1 / successful_predictions
       avg_error_p2 = total_error_p2 / successful_predictions
       std_p1 = np.std(errors_p1)
       std_p2 = np.std(errors_p2)
       
       print(f"\nFinal Results:")
       print(f"Point 1 (x1,y1):")
       print(f"  Average distance error: {avg_error_p1:.2f} pixels")
       print(f"  Standard deviation: {std_p1:.2f} pixels")
       print(f"\nPoint 2 (x2,y2):")
       print(f"  Average distance error: {avg_error_p2:.2f} pixels")
       print(f"  Standard deviation: {std_p2:.2f} pixels")
       print(f"\nOverall:")
       print(f"  Average distance error: {(avg_error_p1 + avg_error_p2)/2:.2f} pixels")
       print(f"  Average standard deviation: {(std_p1 + std_p2)/2:.2f} pixels")
       print(f"  Successful predictions: {successful_predictions}")
   print(f"Failed predictions at indices: {failed_predictions}")
   print(f"Number of failures: {len(failed_predictions)}")

# First do the dataset loading and splitting
split_ratio = 0.9
dataset_dict = load_dataset("jwaters8978/web_scraper_dataset", name="default")
dataset = dataset_dict['train']
#dataset = dataset.select(range(100))

# First split into train and temp
temp_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_data = temp_dataset['train']  # 80%

# Second split: Split the temp into validation and test
val_test_dataset = temp_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
val_data = val_test_dataset['train']    # 10%
test_data = val_test_dataset['test']    # 10%

print("ANALYZING IMAGE ACCURACY")

# Main execution
manager = ModelMemoryManager()
with manager.load_model(
   MllamaForConditionalGeneration,
   "meta-llama/Llama-3.2-11B-Vision-Instruct",
   MllamaProcessor,
   peft_model_path="finetuned_model/fine-tuned/peft_weights/",
   torch_dtype=torch.bfloat16,
   device_map="auto",
   use_safetensors=True
) as (model, processor):
   # Process images in batches using test_data
   # Set show_images=True if you want to see the visualizations
   process_batch(model, processor, test_data, start_index=0, batch_size=180, show_images=False)