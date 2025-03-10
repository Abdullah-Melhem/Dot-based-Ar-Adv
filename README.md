#Adversarial Text Attack in Arabic Sentiment Analysis
##Overview
This project implements an adversarial attack on Arabic sentiment analysis models by generating adversarial text examples using visually similar Arabic characters. The attack modifies key words in a sentence to evaluate the robustness of sentiment classifiers.

Features
Text Preprocessing: Cleans and tokenizes Arabic text.
Sentiment Analysis: Uses a transformer-based model for sentiment prediction.
Adversarial Attacks: Identifies the most vulnerable words and replaces them with visually similar alternatives.
Evaluation: Measures sentiment score changes before and after the attack.
Dependencies
Ensure you have the following Python packages installed:
