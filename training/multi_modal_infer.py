import argparse
import os
import sys
import torch
from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from peft import PeftModel

accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"

def load_model_and_processor(model_name: str, peft_model_path: str = None):
    """
    Load the model and processor based on the 11B or 90B model.
    If peft_model_path is provided, loads the LoRA weights on top.
    """
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    
    # Load PEFT/LoRA weights if path is provided
    if peft_model_path:
        print(f"Loading PEFT model from {peft_model_path}")
        model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
    processor = MllamaProcessor.from_pretrained(model_name, use_safetensors=True)
    model, processor = accelerator.prepare(model, processor)
    return model, processor

def process_image(image_path: str) -> PIL_Image.Image:
    """
    Open and convert an image from the specified path.
    """
    if not os.path.exists(image_path):
        print(f"The image file '{image_path}' does not exist.")
        sys.exit(1)
        
    try:
        with open(image_path, "rb") as f:
            image = PIL_Image.open(f).convert("RGB")
            print(f"Successfully loaded image from {image_path}")
            return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        sys.exit(1)

def generate_text_from_image(
    model, 
    processor, 
    image, 
    prompt_text: str, 
    temperature: float, 
    top_p: float,
    max_new_tokens: int = 512
):
    """
    Generate text from an image using the model and processor.
    """
    try:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
        ]
        
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        
        inputs = processor(image, prompt, return_tensors="pt").to(device)
        
        print("Generating response...")
        output = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        
        generated_text = processor.decode(output[0])[len(prompt):]
        return generated_text
    
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise

def main(
    image_path: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    model_name: str,
    peft_model_path: str = None,
    max_new_tokens: int = 512
):
    """
    Main function to handle the image-to-text generation pipeline.
    """
    try:
        print("Loading model and processor...")
        model, processor = load_model_and_processor(model_name, peft_model_path)
        
        print("Processing image...")
        image = process_image(image_path)
        
        print("Generating text from image...")
        result = generate_text_from_image(
            model,
            processor,
            image,
            prompt_text,
            temperature,
            top_p,
            max_new_tokens
        )
        
        print("\nGenerated Text:")
        print("-" * 50)
        print(result.strip())
        print("-" * 50)

        return result
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from an image and prompt using the Llama Vision model with optional LoRA weights."
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file"
    )
    
    parser.add_argument(
        "--prompt_text",
        type=str,
        required=True,
        help="Prompt text to guide the image description"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for generation (default: 0.9)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Base model name (default: '{DEFAULT_MODEL}')"
    )
    
    parser.add_argument(
        "--peft_model_path",
        type=str,
        help="Path to the PEFT/LoRA model directory containing adapter weights"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    
    args = parser.parse_args()
    
    print("Starting image-to-text generation...")
    print(f"Using model: {args.model_name}")
    if args.peft_model_path:
        print(f"Using PEFT/LoRA from: {args.peft_model_path}")
    
    main(
        args.image_path,
        args.prompt_text,
        args.temperature,
        args.top_p,
        args.model_name,
        args.peft_model_path,
        args.max_new_tokens
    )