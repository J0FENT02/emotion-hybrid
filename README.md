# Hybrid Emotion Detection

## Overview
Combines NRC lexicon + DistilRoBERTa features in a logistic regression pipeline.

## Setup
```bash
git clone https://github.com/yourusername/emotion-hybrid.git
cd emotion-hybrid
python3 -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt

## Data
The “Emotion Dataset for Emotion Recognition Tasks” from Kaggle is used.
https://www.kaggle.com/datasets/parulpandey/emotion-dataset
Download this dataset and rename to "emotion-dataset.csv" as used in the code.
