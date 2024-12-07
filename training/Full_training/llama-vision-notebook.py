{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Llama 3.2 Vision Fine-tuning for Button Detection\n",
    "\n",
    "This notebook implements fine-tuning of Llama 3.2 Vision model for detecting and describing buttons in UI images using the Wave UI dataset.\n",
    "\n",
    "## Setup and Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install required packages\n",
    "!pip install transformers torch datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import necessary libraries\n",
    "import torch\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer\n",
    "from PIL import Image\n",
    "import json\n",
    "import os\n",
    "from datasets import load_dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset Class Implementation\n",
    "\n",
    "This class handles the processing of the Wave UI dataset, including all button parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "class ButtonDetectionDataset(Dataset):\n",
    "    def __init__(self, dataset, processor):\n",
    "        self.dataset = dataset\n",
    "        self.processor = processor\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.dataset)\n",
    "    \n",
    "    def format_button_info(self, button_data):\n",
    "        \"\"\"Format button information into a structured description\"\"\"\n",
    "        bbox = button_data['bbox']\n",
    "        return (f\"Button '{button_data['name']}' of type '{button_data['type']}' \"\n",
    "                f\"is located at coordinates x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}. \"\n",
    "                f\"Purpose: {button_data['purpose']}. \"\n",
    "                f\"Description: {button_data['description']}. \"\n",
    "                f\"Expected behavior: {button_data['expectation']}. \"\n",
    "                f\"Resolution: {button_data['resolution']}.\")\n",
    "    \n",
    "    def create_instruction_prompt(self, instruction):\n",
    "        \"\"\"Create a specific instruction prompt based on the dataset instruction\"\"\"\n",
    "        return f\"Following the instruction '{instruction}', analyze this image and provide detailed information about the button.\"\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        item = self.dataset[idx]\n",
    "        image = Image.open(item['image']).convert('RGB')\n",
    "        \n",
    "        messages = [\n",
    "            {\n",
    "                \"role\": \"user\",\n",
    "                \"content\": [\n",
    "                    {\"type\": \"image\"},\n",
    "                    {\"type\": \"text\", \"text\": self.create_instruction_prompt(item['instruction'])}\n",
    "                ]\n",
    "            }\n",
    "        ]\n",
    "        \n",
    "        target_text = self.format_button_info(item)\n",
    "        \n",
    "        inputs = self.processor(\n",
    "            images=image,\n",
    "            text=self.processor.apply_chat_template(messages, add_generation_prompt=True),\n",
    "            return_tensors=\"pt\",\n",
    "            padding=True,\n",
    "            truncation=True\n",
    "        )\n",
    "        \n",
    "        target_inputs = self.processor(\n",
    "            text=target_text,\n",
    "            return_tensors=\"pt\",\n",
    "            padding=True,\n",
    "            truncation=True\n",
    "        )\n",
    "        \n",
    "        for k, v in inputs.items():\n",
    "            inputs[k] = v.squeeze(0)\n",
    "        \n",
    "        inputs[\"labels\"] = target_inputs.input_ids.squeeze(0)\n",
    "        \n",
    "        return inputs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model and Data Loading\n",
    "\n",
    "Load the Llama model and the Wave UI dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load model and processor\n",
    "model_id = \"/kaggle/input/llama-3.2-vision/transformers/11b-vision-instruct/1\"\n",
    "processor = AutoProcessor.from_pretrained(model_id)\n",
    "model = AutoModelForCausalLM.from_pretrained(model_id)\n",
    "\n",
    "# Load dataset\n",
    "dataset = load_dataset(\"agentsea/wave-ui-25k\")\n",
    "print(f\"Dataset loaded with {len(dataset['train'])} training samples and {len(dataset['validation'])} validation samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prepare Datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Create train and validation datasets\n",
    "train_dataset = ButtonDetectionDataset(dataset['train'], processor)\n",
    "val_dataset = ButtonDetectionDataset(dataset['validation'], processor)\n",
    "\n",
    "# Check a sample input\n",
    "sample = train_dataset[0]\n",
    "print(\"Sample input shape:\", {k: v.shape for k, v in sample.items() if isinstance(v, torch.Tensor)})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Define training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=\"./button-detection-model\",\n",
    "    num_train_epochs=3,\n",
    "    per_device_train_batch_size=2,\n",
    "    per_device_eval_batch_size=2,\n",
    "    warmup_steps=500,\n",
    "    weight_decay=0.01,\n",
    "    logging_dir=\"./logs\",\n",
    "    logging_steps=10,\n",
    "    evaluation_strategy=\"steps\",\n",
    "    eval_steps=100,\n",
    "    save_steps=100,\n",
    "    gradient_accumulation_steps=8,\n",
    "    fp16=True,\n",
    "    load_best_model_at_end=True,\n",
    "    learning_rate=2e-5,\n",
    "    lr_scheduler_type=\"cosine\",\n",
    "    metric_for_best_model=\"eval_loss\",\n",
    "    greater_is_better=False,\n",
    "    early_stopping_patience=3\n",
    ")\n",
    "\n",
    "# Initialize trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=train_dataset,\n",
    "    eval_dataset=val_dataset,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Start Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Train the model\n",
    "trainer.train()\n",
    "\n",
    "# Save the model\n",
    "trainer.save_model(\"./button-detection-model-final\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prediction and Evaluation Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def predict_button_details(model, processor, image_path, instruction=\"Describe the button in this image\"):\n",
    "    \"\"\"Predict button details for a new image\"\"\"\n",
    "    image = Image.open(image_path).convert('RGB')\n",
    "    \n",
    "    messages = [\n",
    "        {\n",
    "            \"role\": \"user\",\n",
    "            \"content\": [\n",
    "                {\"type\": \"image\"},\n",
    "                {\"type\": \"text\", \"text\": instruction}\n",
    "            ]\n",
    "        }\n",
    "    ]\n",
    "    \n",
    "    inputs = processor(\n",
    "        images=image,\n",
    "        text=processor.apply_chat_template(messages, add_generation_prompt=True),\n",
    "        return_tensors=\"pt\",\n",
    "    )\n",
    "    \n",
    "    outputs = model.generate(\n",
    "        **inputs,\n",
    "        max_new_tokens=200,\n",
    "        do_sample=True,\n",
    "        temperature=0.7,\n",
    "        top_p=0.9,\n",
    "        num_return_sequences=1\n",
    "    )\n",
    "    \n",
    "    return processor.decode(outputs[0], skip_special_tokens=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "def evaluate_predictions(model, processor, test_dataset, num_samples=10):\n",
    "    \"\"\"Evaluate model predictions\"\"\"\n",
    "    results = []\n",
    "    for i in range(num_samples):\n",
    "        sample = test_dataset[i]\n",
    "        prediction = predict_button_details(\n",
    "            model, \n",
    "            processor, \n",
    "            sample['image'], \n",
    "            sample['instruction']\n",
    "        )\n",
    "        \n",
    "        results.append({\n",
    "            'instruction': sample['instruction'],\n",
    "            'ground_truth': {\n",
    "                'name': sample['name'],\n",
    "                'type': sample['type'],\n",
    "                'bbox': sample['bbox'],\n",
    "                'purpose': sample['purpose'],\n",
    "                'description': sample['description'],\n",
    "                'expectation': sample['expectation'],\n",
    "                'resolution': sample['resolution']\n",
    "            },\n",
    "            'prediction': prediction\n",
    "        })\n",
    "    \n",
    "    return results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test the Model\n",
    "\n",
    "Try the model on some test images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Evaluate on test samples\n",
    "test_results = evaluate_predictions(model, processor, dataset['validation'], num_samples=5)\n",
    "\n",
    "# Display results\n",
    "for idx, result in enumerate(test_results):\n",
    "    print(f\"\\nTest Sample {idx + 1}\")\n",
    "    print(\"Instruction:\", result['instruction'])\n",
    "    print(\"\\nPrediction:\", result['prediction'])\n",
    "    print(\"\\nGround Truth:\")\n",
    "    for key, value in result['ground_truth'].items():\n",
    "        print(f\"{key}: {value}\")\n",
    "    print(\"-\" * 80)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
