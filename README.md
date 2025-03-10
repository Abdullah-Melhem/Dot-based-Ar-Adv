# Arabic Text Adversarial Attack

This project implements an adversarial attack on Arabic text using NLP techniques. It leverages **transformers**, **NLTK**, **SentenceTransformers**, and **pandas** to manipulate and analyze text data. The main goal is to identify the most vulnerable words in a sentence and generate adversarial examples by substituting visually similar Arabic characters.

## Features

- **Text Cleaning**: Removes punctuation, emojis, stopwords, and noise from Arabic text.
- **Word Tokenization**: Splits text into individual words.
- **Most Important Word (MIW) Identification**: Finds the most influential words in a sentence.
- **Adversarial Word Generation**: Replaces MIWs with visually similar Arabic characters.
- **Sentiment Analysis**: Computes sentiment scores before and after attack.
- **Attack Simulation**: Applies the adversarial attack step-by-step to analyze the impact.

## Installation

Before running the script, install the required dependencies:

```bash
pip install torch transformers pandas tqdm nltk sentence-transformers scikit-learn tensorflow huggingface-hub
