# Auto-Review Analyzer: AI-Powered Pros & Cons Extraction and Summarization  
**A scalable NLP system for automatically identifying, classifying, and summarizing positive and negative reviews using Transformer architectures and LoRA fine-tuning**

This project develops an end-to-end NLP system that automatically extracts and summarizes pros and cons from user-generated reviews. By combining a custom Transformer classifier with LoRA fine-tuning and BART-large summarization, it transforms unstructured opinion text into actionable insights for product analysis, feedback aggregation, and decision support.

## Overview
Using the Glassdoor job reviews dataset, this project implements a full NLP pipeline combining custom model architecture, parameter-efficient fine-tuning, and production-ready deployment. Key achievements include:

- **High-accuracy sentence classification** with a custom Transformer encoder achieving 94.60% accuracy on held-out test data
- **Parameter-efficient adaptation** using LoRA, reducing trainable parameters to 0.03% while improving performance
- **Complete processing pipeline** from raw text to structured summaries via sentence splitting, classification, aggregation, and summarization
- **Interactive deployment** with a Gradio web interface for real-time analysis and visualization

## Key Components
- **Data preprocessing & feature engineering** with custom text cleaning and GloVe-based vocabulary building
- **Custom Transformer architecture** with positional encoding and masked mean pooling
- **Parameter-efficient fine-tuning** using LoRA (Low-Rank Adaptation) for efficient model adaptation
- **Sentence classification** into Pro/Con categories at the sentence level
- **Text summarization** with BART-large for generating concise pros and cons summaries
- **Interactive deployment** via Gradio with color-coded visualization and example reviews

## Data
- **Primary dataset**: Glassdoor job reviews (~1.7 million sentences) with explicit pros/cons labeling
- **Embeddings**: GloVe 6B 100-dimensional pre-trained word vectors
- **Preprocessing**: Custom cleaning pipeline removing pronouns, special characters, and irrelevant content

## Models & Metrics
| Task | Model | Description |
|------|-------|-------------|
| Sentence Classification | Custom Transformer Encoder + LoRA | Binary classification (Pro/Con) with GloVe embeddings and positional encoding |
| Fine-tuning | LoRA (Low-Rank Adaptation) | Parameter-efficient adaptation with only 12,354 trainable parameters |
| Summarization | BART-large-cnn-samsum | Concise summary generation for grouped pros and cons |
| Tokenization | TorchText + NLTK | Sentence segmentation and vocabulary building from GloVe |

**Metrics**: 94.60% classification accuracy, 0.03% trainable parameters via LoRA, real-time inference capability

## Workflow
1. **Data preparation**: Load and preprocess Glassdoor reviews, build vocabulary from GloVe embeddings
2. **Model development**: Implement custom Transformer encoder
3. **Training**: Fine-tune with LoRA using mixed-precision training for efficiency
4. **Pipeline integration**: Combine classifier with sentence splitting and BART summarizer
5. **Evaluation**: Test on held-out data and example reviews for qualitative assessment
6. **Deployment**: Build interactive Gradio interface for real-time text analysis

## Results
- **Classification accuracy**: 94.60% on test set (LoRA fine-tuned model)
- **Efficiency**: LoRA adaptation reduces trainable parameters to 12,354 (0.03% of total)
- **Processing capability**: Handles sequences up to 3,379 tokens with dynamic padding
- **Usability**: Interactive web interface with color-coded visualization and example reviews
- **Scalability**: Modular design supporting extension to other domains and languages

## Installation
pip install -r requirements.txt
