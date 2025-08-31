# VLM Implementation

This repository contains my implementation of a Vision-Language Model (VLM) training pipeline, supporting both pre-training and supervised fine-tuning stages.

## Overview

I developed a complete VLM training system that integrates vision and language modalities. The implementation includes:

- **Model Architecture**: Combines CLIP vision encoder with a language model
- **Training Pipeline**: Both pre-training and supervised fine-tuning capabilities
- **Evaluation Framework**: Comprehensive evaluation metrics for model performance

## Features

- **Dual Training Stages**: Pre-training and supervised fine-tuning
- **Efficient Data Handling**: Supports JSONL datasets with corresponding image data
- **Training Monitoring**: Optional Weights & Biases integration for experiment tracking
- **Flexible Configuration**: Adjustable model dimensions and training parameters

## Technical Implementation

### Model Components
- **Vision Encoder**: CLIP ViT-Base-Patch16 for image feature extraction
- **Language Model**: Transformer-based architecture for text generation
- **Multimodal Fusion**: Attention mechanisms for vision-language alignment

### Training Process
1. **Pre-training**: Learn visual-linguistic representations from paired data
2. **Fine-tuning**: Task-specific adaptation using supervised learning
3. **Checkpointing**: Automatic weight saving every 100 steps

## Requirements

- Python 3.10.16
- CUDA 12.2
- Dependencies specified in requirements.txt

## Usage

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights
git clone https://huggingface.co/openai/clip-vit-base-patch16 ./model/vision_model/
