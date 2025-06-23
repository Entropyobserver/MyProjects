# Sentiment Analysis with Transformers

## Project Overview

This project implements a text sentiment classification model using Transformer-based architectures, specifically BERT. The model predicts whether a given English sentence expresses positive or negative sentiment. It is deployed on Hugging Face Spaces using a Gradio interface.

This end-to-end solution includes preprocessing, fine-tuning, and real-time inference.

## Live Demo

Access the live demo here:  
https://huggingface.co/spaces/entropy25/sentiment-analysis

## Project Structure

- `notebooks/` - Jupyter Notebook for data processing, model training, and evaluation
- `app/` - Gradio application script used for Hugging Face Spaces deployment
- `requirements.txt` - Python dependencies
- `README.md` - Project description

## Technologies Used

- Model: BERT-base-uncased
- Libraries: Hugging Face Transformers, PyTorch, Gradio
- Deployment: Hugging Face Spaces

## Performance

- Accuracy: Approximately 93 percent on standard sentiment classification datasets such as IMDb
- Inference: Real-time with confidence scores

## Example

Input:  
`I absolutely loved this movie. The acting was fantastic.`

Output:  
`Sentiment: POSITIVE`  
`Confidence: 97.3%`

## How to Run Locally

```bash
git clone https://github.com/Entropyobserver/sentiment-analysis-transformers
cd sentiment-analysis-transformers
pip install -r requirements.txt
python app/app.py
