from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel, AutoModelForCausalLM
from datasets import load_dataset
import torch
from PIL import Image
import requests
from io import BytesIO

from huggingface_hub import login
login(token='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')

# Load the Wave-UI-25k dataset
dataset = load_dataset("miketes/Web-filtered-english-wave-ui-25k")

# Load the LLaMA 3 model and tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
llama_model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B")

# Load CLIP (or another image encoder) for converting images to embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the bounding box embedding layer
bbox_embedding_layer = nn.Linear(4, llama_model.config.hidden_size)