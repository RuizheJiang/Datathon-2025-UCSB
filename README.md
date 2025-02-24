# Clash Royale S18 Card Deck Analysis

## Overview
Clash Royale is a popular mobile game. It features card decks and online matching, so it may generate some interesting data for us to analyze. 
Hence, we obtained 3.7 million Clash Royale match results. Our goal is to predict the win rate of a match based on the decks of the players and to analyze
the game environment based on popular card decks as well as match win rates. The script folder contains the code we wrote to generate the results, and the 
report is saved in Google slides. 

## Datasets
We use public datasets from hugging face and kaggle.
- https://www.kaggle.com/datasets/tristanwassner/clash-royale-s18-ladder-datasets-for-prediction/
- https://huggingface.co/datasets/Grandediw/clash-royale-battle

## How to run this project
'''bash
python train.py
'''

## Features
- We found popular card decks using a clustering algorithm. 
- Matches are also categorized into clusters. 
- We developed a three-layer attention architecture that leverages one-hot encoded card features to capture the intricate interplay between cards in a deck.

## Future Work
We plan to explore more advanced multi-head, multi-layer attention mechanisms tailored to this task to further enhance performance. Additionally, we aim to deepen our research into model interpretability, striving to provide comprehensive explanations both at a local (individual prediction) level and globally, to fully understand and articulate how our model makes its decisions.
