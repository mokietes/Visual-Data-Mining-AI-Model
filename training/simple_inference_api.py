import torch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from PIL import Image
from accelerate import Accelerator

# Global variables to maintain model state
_model = None
_processor = None
_device = Accelerator().device

def generate_caption(image_path: str, prompt: str) -> str:
    """
    Generate a caption for an image given a prompt.
    """
    global _model, _processor
    
    # Lazy load model if needed
    if _model is None:
        _model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map=_device
        )
        _processor = MllamaProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    # Process image and generate caption
    image = Image.open(image_path).convert('RGB')
    
    # The processor expects a string prompt, not a dict
    inputs = _processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(_device)
    
    with torch.no_grad():
        output = _model.generate(**inputs, max_new_tokens=512)
        
    return _processor.decode(output[0], skip_special_tokens=True)