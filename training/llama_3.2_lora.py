import os
import torch
import logging
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfFolder

# Constants
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = "llama-vision-ui-checkpoint-full"
DATASET_NAME = "miketes/Web-filtered-english-wave-ui-25k"

def setup_model_and_processor(checkpoint_dir=None):
    """Setup model optimized for A100 GPU"""
    print("Loading model and processor...")
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID if checkpoint_dir is None else checkpoint_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    return model, processor

def prepare_dataset(dataset, processor):
    """Convert dataset to format expected by model and tokenize"""
    print(f"Preparing dataset with {len(dataset)} examples...")
    
    def format_instruction(example):
        instruction = f"""Return the bounding box of the {example['description']}. It's used to {example['purpose']} and if we click it {example['expectation']}."""
        bbox = example['bbox']
        x_res, y_res = example['resolution']
        #bounding boxes are in percentage of the screen
        bbox[0] = bbox[0] / x_res * 100
        bbox[1] = bbox[1] / y_res * 100
        bbox[2] = bbox[2] / x_res * 100
        bbox[3] = bbox[3] / y_res * 100
        bbox = [round(x, 2) for x in bbox]  # Limit to two decimal places
        print(bbox)
        reply = f"""{str(bbox)}"""
        
        # Create messages format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example['image']},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reply}]
            }
        ]
        
        # Tokenize the messages
        texts = processor.apply_chat_template(messages, tokenize=False)
        inputs = processor(
            text=texts,
            images=example['image'],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Remove the batch dimension since we're processing one example at a time
        return {k: v.squeeze(0) for k, v in inputs.items()}

    prepared_data = []
    for example in dataset:
        try:
            formatted = format_instruction(example)
            prepared_data.append(formatted)
        except Exception as e:
            print(f"Skipping example due to error: {e}")
            continue

    print(f"Dataset preparation complete. Formatted {len(prepared_data)} examples.")
    return prepared_data

def create_data_collator(processor):
    """Create a data collator that handles images and text"""
    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) 
                for example in examples]
        images = [example["messages"][0]["content"][0]["image"] 
                 for example in examples]
        
        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        
        batch["labels"] = labels
        return batch
    
    return collate_fn

def train(resume_from_checkpoint=None):
    """Main training function"""
    print("Starting training setup...")
    
    model, processor = setup_model_and_processor(resume_from_checkpoint)
    
    print(f"Loading dataset from {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    MAX_SAMPLES = 10  # testing
    dataset = dataset.select(range(min(len(dataset), MAX_SAMPLES)))
    print(f"Dataset size: {len(dataset)} samples")
    
    # Pass processor to prepare_dataset
    prepared_dataset = prepare_dataset(dataset, processor)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=1e-5,
        bf16=True,
        tf32=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        report_to="none",  # Changed from "tensorboard" to "none"
        remove_unused_columns=False,
        push_to_hub=False,
        max_steps=5000,
        dataloader_num_workers=4,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16_full_eval=True,
        ddp_find_unused_parameters=False,
        group_by_length=True,
    )

    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=create_data_collator(processor),
        tokenizer=processor.tokenizer,
        dataset_kwargs={'skip_prepare_dataset': True}  
    )
    
    print("\nTraining parameters:")
    print(f"Number of training examples: {len(prepared_dataset)}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log'))
        ]
    )
    
    token = os.getenv('HF_TOKEN')
    if token:
        from huggingface_hub import login
        login(token=token)
    else:
        if HfFolder.get_token() is not None:
            print("Using cached Hugging Face token")
        else:
            print("No token found. Please set HF_TOKEN environment variable or login manually first")
    
    checkpoint_path = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(OUTPUT_DIR, latest_checkpoint)
            print(f"Found checkpoint: {checkpoint_path}")
    
    try:
        train(resume_from_checkpoint=checkpoint_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise
    finally:
        print("\nCleaning up...")
