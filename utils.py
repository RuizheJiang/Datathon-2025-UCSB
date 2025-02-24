import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
# import kaggle
from datasets import load_dataset
import copy
import torch
from sklearn.cluster import KMeans
import kmedoids
from sklearn.metrics import pairwise_distances

import os
import matplotlib.pyplot as plt
import cv2
from difflib import get_close_matches

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def getMatches(df, total_num_cards, id_to_id):
  matches = []
  for row in df.iloc:
    match1 = Match(row, total_num_cards, id_to_id)
    match2 = copy.deepcopy(match1)
    match2.winner_info, match2.loser_info = match2.loser_info, match2.winner_info
    
    matches.append([match1, 1])
    matches.append([match2, 0])
  return matches


def draw_deck(card_names, folder_path):
    images = []
    for i in range(len(card_names)):
        image_path = os.path.join(folder_path, card_names[i])+".png"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if images[i] is not None:
            ax.imshow(images[i])
            ax.set_title(card_names[i])
        else:
            ax.set_title("Not Found")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()



def draw_decks(all_decks, folder_path, x, y):
    fig, axes = plt.subplots(x, y, figsize=(20, 10))
    
    for idx, ax in enumerate(axes.flat):
        card_names = all_decks[idx]
        images = []
        for name in card_names:
            image_path = os.path.join(folder_path, name) + ".png"
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        deck_grid = np.ones((200, 400, 3), dtype=np.uint8) * 255
        row_imgs = []
        for i in range(0, len(images), 4):
            row = images[i:i+4]
            row = [
                cv2.resize(img, (70, 100)) if img is not None else np.ones((100, 70, 3), dtype=np.uint8) * 255
                for img in row
            ]
            row_concat = np.hstack(row)
            row_imgs.append(row_concat)
        if row_imgs:
            deck_img = np.vstack(row_imgs)
            ax.imshow(deck_img)
        ax.set_title(f"Deck {idx+1}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()



class Match:
  def __init__(self, row, total_num_cards, id_to_id):
    self.winner_card = np.zeros((total_num_cards))
    self.loser_card = np.zeros((total_num_cards))

    winner_card_ids = []
    for card_id in row["winner.cards.list"][1:-1].split(", "):
        winner_card_ids.append(id_to_id[int(card_id)])
    loser_card_ids = []
    for card_id in row["loser.cards.list"][1:-1].split(", "):
        loser_card_ids.append(id_to_id[int(card_id)])

    self.winner_card[winner_card_ids] = 1
    self.loser_card[loser_card_ids] = 1

    self.winner_info = {"cards": self.winner_card, 
                        "Trophies": row["winner.startingTrophies"].item(), 
                        "troop_count": row["winner.troop.count"].item(),
                        "structure_count": row["winner.structure.count"].item(),
                        "spell_count": row["winner.spell.count"].item(),
                        "elixir_average": row["winner.elixir.average"].item()}
    self.loser_info = {"cards": self.loser_card, 
                        "Trophies": row["loser.startingTrophies"].item(), 
                        "troop_count": row["loser.troop.count"].item(),
                        "structure_count": row["loser.structure.count"].item(),
                        "spell_count": row["loser.spell.count"].item(),
                        "elixir_average": row["loser.elixir.average"].item()}


def plot_and_save(train_loss, train_acc, filename='training_plot.png'):
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss (Left Y-axis)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:red', marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot Accuracy (Right Y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Train Accuracy (%)', color='tab:blue')
    ax2.plot(epochs, train_acc, color='tab:blue', marker='s', label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title and Layout
    plt.title('Training Loss & Accuracy over Epochs')
    fig.tight_layout()

    # Save to file
    plt.savefig(filename)
    plt.close()
