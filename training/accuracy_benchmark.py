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
import argparse
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
import torch.distributed as dist
import torch.nn as nn

class ModelMemoryManager:
    def _consolidate_fsdp_checkpoint(self, checkpoint_path):
        """Consolidate FSDP sharded checkpoints into a full model checkpoint"""
        print(f"Consolidating FSDP checkpoint from {checkpoint_path}")
        
        try:
            # Initialize distributed environment for single GPU consolidation
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='tcp://localhost:29500',
                    world_size=1,
                    rank=0
                )
            
            # Create a dummy model to handle FSDP state dict
            dummy_model = FSDP(torch.nn.Linear(1, 1))
            
            # Configure FSDP state dict settings
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True
            )
            
            with dummy_model.state_dict_type(
                dummy_model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=full_state_dict_config
            ):
                # Load and consolidate the sharded checkpoint
                checkpoint_files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.distcp')])
                consolidated_state = {}
                
                for file in checkpoint_files:
                    shard_path = os.path.join(checkpoint_path, file)
                    print(f"Loading shard: {shard_path}")
                    shard_state = torch.load(shard_path, map_location='cpu')
                    consolidated_state.update(shard_state)

                # Save consolidated checkpoint
                consolidated_path = os.path.join(checkpoint_path, "consolidated_model.pt")
                torch.save(consolidated_state, consolidated_path)
                print(f"Saved consolidated checkpoint to {consolidated_path}")
                return consolidated_path

        except Exception as e:
            print(f"Error consolidating FSDP checkpoint: {str(e)}")
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
                
    def _is_fsdp_checkpoint(self, checkpoint_path):
        """Check if directory contains FSDP checkpoint files"""
        if not os.path.exists(checkpoint_path):
            return False
        files = os.listdir(checkpoint_path)
        return any(f.endswith('.distcp') for f in files)

    @contextmanager
    def load_model(self, model_class, model_name, processor_class=None, checkpoint_path=None, peft_model_path=None, model_type='full', **kwargs):
        model = None
        processor = None
        try:
            if model_type == 'full':
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    raise ValueError(f"Full model checkpoint not found at {checkpoint_path}")
                
                if self._is_fsdp_checkpoint(checkpoint_path):
                    print("Found FSDP checkpoint, consolidating...")
                    consolidated_path = self._consolidate_fsdp_checkpoint(checkpoint_path)
                    print(f"Loading consolidated model from: {consolidated_path}")
                    model = model_class.from_pretrained(
                        model_name,
                        state_dict=torch.load(consolidated_path),
                        **kwargs
                    )
                else:
                    print(f"Loading standard checkpoint from: {checkpoint_path}")
                    model = model_class.from_pretrained(
                        checkpoint_path,
                        local_files_only=True,
                        **kwargs
                    )
            elif model_type == 'peft':
                if not peft_model_path or not os.path.exists(peft_model_path):
                    raise ValueError(f"PEFT checkpoint not found at {peft_model_path}")
                print(f"Loading base model: {model_name}")
                model = model_class.from_pretrained(model_name, **kwargs)
                print(f"Applying PEFT weights from: {peft_model_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, peft_model_path)
            else:
                raise ValueError(f"Invalid model_type: {model_type}. Must be 'full' or 'peft'")
            
            print("Model loaded successfully")
            processor = processor_class.from_pretrained(model_name) if processor_class else None
            yield model, processor
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        finally:
            if model:
                del model
            if processor:
                del processor
            torch.cuda.empty_cache()
            gc.collect()

def convert_to_pixels(bbox, image_size):
    """Convert normalized bbox coordinates to pixel coordinates"""
    x_res, y_res = image_size
    return [
        bbox[0] * x_res / 100,
        bbox[1] * y_res / 100,
        bbox[2] * x_res / 100,
        bbox[3] * y_res / 100
    ]

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_batch(model, processor, dataset, start_index, batch_size, show_images=False):
    """Process a batch of images and compute metrics"""
    total_error_p1 = 0
    total_error_p2 = 0
    errors_p1 = []
    errors_p2 = []
    successful_predictions = 0
    failed_predictions = []
    
    for i in range(batch_size):
        index = start_index + i
        image = dataset[index]['images']
        prompt = dataset[index]['texts'][0]['user']
        
        # Process image
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
        
        # Generate prediction
        output = model.generate(
            **inputs,
            temperature=0.5,
            top_p=0.8,
            max_new_tokens=512,
        )
        pred = processor.decode(output[0])[len(prompt_text):]
        
        # Process ground truth
        true_box = [float(x) for x in dataset[index]['texts'][0]['assistant'].strip('[]').split(',')]
        true_box_pixels = convert_to_pixels(true_box, image.size)
        
        try:
            # Extract predicted coordinates
            pred_numbers = pred.split('|>')[1].split('<|')[0].strip()
            pred_box = [float(x) for x in pred_numbers.strip('[]').split(',')]
            pred_box_pixels = convert_to_pixels(pred_box, image.size)
            
            # Calculate metrics
            p1_distance = calculate_distance(
                [true_box_pixels[0], true_box_pixels[1]], 
                [pred_box_pixels[0], pred_box_pixels[1]]
            )
            p2_distance = calculate_distance(
                [true_box_pixels[2], true_box_pixels[3]], 
                [pred_box_pixels[2], pred_box_pixels[3]]
            )
            
            # Update statistics
            total_error_p1 += p1_distance
            total_error_p2 += p2_distance
            errors_p1.append(p1_distance)
            errors_p2.append(p2_distance)
            successful_predictions += 1
            
            # Visualize if requested
            if show_images:
                draw = ImageDraw.Draw(image)
                draw.rectangle(true_box_pixels, outline='green', width=3)
                draw.rectangle(pred_box_pixels, outline='red', width=3)
                plt.figure(figsize=(12,8))
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                plt.close()
                
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model_type', type=str, choices=['full', 'peft'], 
                      required=True, help='Type of model to evaluate (full or peft)')
    parser.add_argument('--batch_size', type=int, default=180,
                      help='Number of images to process')
    parser.add_argument('--show_images', action='store_true',
                      help='Show visualization of predictions')
    args = parser.parse_args()

    # Load and split dataset
    dataset_dict = load_dataset("jwaters8978/web_scraper_dataset_2", name="default")
    dataset = dataset_dict['train']
    
    # Split dataset
    temp_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_data = temp_dataset['train']  # 80%
    
    val_test_dataset = temp_dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
    val_data = val_test_dataset['train']    # 10%
    test_data = val_test_dataset['test']    # 10%
    
    print(f"ANALYZING IMAGE ACCURACY USING {args.model_type.upper()} MODEL")
    
    # Set up checkpoint paths
    full_checkpoint_path = "trained_model/full_training-meta-llama/Llama-3.2-11B-Vision-Instruct"
    peft_checkpoint_path = "finetuned_model/fine-tuned/peft_weights"
    
    # Print checkpoint information
    print(f"\nCheckpoint paths:")
    if args.model_type == 'full':
        print(f"Using full checkpoint path: {full_checkpoint_path}")
        if os.path.exists(full_checkpoint_path):
            print(f"Checkpoint contents: {os.listdir(full_checkpoint_path)}")
    else:
        print(f"Using PEFT checkpoint path: {peft_checkpoint_path}")
        if os.path.exists(peft_checkpoint_path):
            print(f"Checkpoint contents: {os.listdir(peft_checkpoint_path)}")
    
    # Load and evaluate model
    manager = ModelMemoryManager()
    with manager.load_model(
        MllamaForConditionalGeneration,
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        MllamaProcessor,
        checkpoint_path=full_checkpoint_path if args.model_type == 'full' else None,
        peft_model_path=peft_checkpoint_path if args.model_type == 'peft' else None,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True
    ) as (model, processor):
        process_batch(model, processor, test_data, 
                     start_index=0, 
                     batch_size=args.batch_size, 
                     show_images=args.show_images)

if __name__ == "__main__":
    main()