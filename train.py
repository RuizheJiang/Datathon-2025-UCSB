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
from utils import *
from model import *
from tqdm import tqdm

higging_face_df = load_dataset("Grandediw/clash-royale-battle")
df = higging_face_df['train'].to_pandas()
all_card_ids = set()
for i in range(8):
  all_card_ids = all_card_ids.union(list(df[f"winner.card{i+1}.id"].unique()))
  all_card_ids = all_card_ids.union(list(df[f"loser.card{i+1}.id"].unique()))
all_card_ids = sorted(list(all_card_ids))
remove_cols = [
 'Unnamed: 0',
 'battleTime',
#  'arena.id',      # Filter
#  'gameMode.id',   # Filter
 # 'average.startingTrophies',  # Filter
 'winner.tag',
#  'winner.startingTrophies',
 'winner.trophyChange',
 'winner.crowns',
 'winner.kingTowerHitPoints',#
 'winner.princessTowersHitPoints',
 'winner.clan.tag',
 'winner.clan.badgeId',
 'loser.tag',
#  'loser.startingTrophies',
 'loser.trophyChange',
 'loser.crowns',
 'loser.kingTowerHitPoints',
 'loser.clan.tag',
 'loser.clan.badgeId', #
 'loser.princessTowersHitPoints',
 'tournamentTag',
 'winner.card1.id',
#  'winner.card1.level',
 'winner.card2.id',
#  'winner.card2.level',
 'winner.card3.id',
#  'winner.card3.level',
 'winner.card4.id',
#  'winner.card4.level',
 'winner.card5.id',
#  'winner.card5.level',
 'winner.card6.id',
#  'winner.card6.level',
 'winner.card7.id',
#  'winner.card7.level',
 'winner.card8.id',
#  'winner.card8.level',
#  'winner.cards.list',
 'winner.totalcard.level',  # TODO
#  'winner.troop.count',
#  'winner.structure.count',
#  'winner.spell.count',
 'winner.common.count',
 'winner.rare.count',
 'winner.epic.count',
 'winner.legendary.count',
#  'winner.elixir.average',
 'loser.card1.id',
#  'loser.card1.level',
 'loser.card2.id',
#  'loser.card2.level',
 'loser.card3.id',
#  'loser.card3.level',
 'loser.card4.id',
#  'loser.card4.level',
 'loser.card5.id',
#  'loser.card5.level',
 'loser.card6.id',
#  'loser.card6.level',
 'loser.card7.id',
#  'loser.card7.level',
 'loser.card8.id',
#  'loser.card8.level',
#  'loser.cards.list',
 'loser.totalcard.level',   # TODO
#  'loser.troop.count',
#  'loser.structure.count',
#  'loser.spell.count',
 'loser.common.count',
 'loser.rare.count',
 'loser.epic.count',
 'loser.legendary.count',
#  'loser.elixir.average'
 ]

df.drop(remove_cols, axis=1, inplace=True)
df.drop(df[df['arena.id'] != 54000050.0].index, inplace=True)
df_mode1 = df.drop(df[df['gameMode.id'] != 72000006.0].index, inplace=False)
df_mode2 = df.drop(df[df['gameMode.id'] != 72000201.0].index, inplace=False)
df_mode1.drop(["arena.id", "gameMode.id"], axis=1, inplace=True)
df_mode2.drop(["arena.id", "gameMode.id"], axis=1, inplace=True)

for i in range(8):
  df_mode1.drop(df_mode1[df_mode1[f"loser.card{i+1}.level"] != 13].index, inplace=True)
  df_mode1.drop(df_mode1[df_mode1[f"winner.card{i+1}.level"] != 13].index, inplace=True)
  df_mode2.drop(df_mode2[df_mode2[f"loser.card{i+1}.level"] != 13].index, inplace=True)
  df_mode2.drop(df_mode2[df_mode2[f"winner.card{i+1}.level"] != 13].index, inplace=True)
  df_mode1.drop([f"loser.card{i+1}.level"], axis=1, inplace=True)
  df_mode1.drop([f"winner.card{i+1}.level"], axis=1, inplace=True)
  df_mode2.drop([f"loser.card{i+1}.level"], axis=1, inplace=True)
  df_mode2.drop([f"winner.card{i+1}.level"], axis=1, inplace=True)

df_mode1 = df_mode1.sort_values(by='average.startingTrophies', ascending=False).head(10000)
cards1 = pd.read_csv("data//Wincons.csv")
cards2 = pd.read_csv("data//CardMasterListSeason18_12082020.csv")
id1 = cards1["card_id"]
names1 = cards1["card_name"]
id2 = cards2["team.card1.id"]
names2 = cards2["team.card1.name"]
id_to_name = {}
for i in range(len(id1)):
    id_to_name[id1[i].item()] = names1[i]
for i in range(len(id2)):
    id_to_name[id2[i].item()] = names2[i]
total_num_cards = len(id_to_name)
id_to_id = {}
id_to_id_reversed = {}
for i, idd in enumerate(id_to_name.keys()):
    id_to_id[idd] = i
    id_to_id_reversed[i] = idd

matches = getMatches(df_mode1, total_num_cards, id_to_id)




def train(model, num_epochs, batch_size, lr, matches, device='cuda'):
    dataset_sampler = MatchSampler(matches)

    train_size = int(0.8 * len(dataset_sampler))
    test_size = len(dataset_sampler) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_sampler, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    train_acc = []
    eval_acc = []

    model = model.to(device)
    best_eval_accuracy = 0
    best_model_path = "predmodel_diff/best_model.pth"

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        model.train()

        for i, data in tqdm(enumerate(train_dataloader)):
            my_cards = torch.tensor(data["my_cards"]).float().to(device)
            opponent_cards = torch.tensor(data["opponent_cards"]).float().to(device)
            my_features = torch.tensor(data["my_features"]).float().to(device)
            opponent_features = torch.tensor(data["opponent_features"]).float().to(device)
            labels = torch.tensor(data["labels"]).to(device)

            optimizer.zero_grad()
            ypred = model(my_cards, opponent_cards, my_features, opponent_features)
            loss = nn.BCELoss()(ypred.squeeze(), labels.float())
            epoch_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()

            predictions = (ypred > 0.5).squeeze()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = 100 * correct / total

        train_loss.append(avg_loss)
        train_acc.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if (epoch + 1) % 2 == 0:
            model.eval()
            eval_correct = 0
            eval_total = 0

            with torch.no_grad():
                for data in test_dataloader:
                    my_cards = torch.tensor(data["my_cards"]).float().to(device)
                    opponent_cards = torch.tensor(data["opponent_cards"]).float().to(device)
                    my_features = torch.tensor(data["my_features"]).float().to(device)
                    opponent_features = torch.tensor(data["opponent_features"]).float().to(device)
                    labels = torch.tensor(data["labels"]).to(device)

                    ypred = model(my_cards, opponent_cards, my_features, opponent_features)
                    predictions = (ypred > 0.5).squeeze()
                    eval_correct += (predictions == labels).sum().item()
                    eval_total += labels.size(0)

            eval_accuracy = 100 * eval_correct / eval_total
            eval_acc.append(eval_accuracy)
            print(f"--- Evaluation after Epoch {epoch+1}: Accuracy: {eval_accuracy:.2f}%")

            # Save best model
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_accuracy': best_eval_accuracy
                }, best_model_path)

    print("Training Completed")
    return train_loss, train_acc, best_eval_accuracy


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def load_checkpoint(model, optimizer, path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    eval_accuracy = checkpoint['eval_accuracy']
    model.to(device)
    return model, optimizer, epoch, eval_accuracy


# model = PredModel_withFeatures(103, 512, 10)
# model = load_model(model, "predmodel_diff/final_model.pth", device='cuda:0')
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# model, optimizer, epoch, best_acc = load_checkpoint(model, optimizer, "predmodel_diff/best_model.pth", device='cuda:0')
# print(f"Resumed from Epoch {epoch} with Best Eval Accuracy: {best_acc:.2f}%")


model = PredModel(103, 512, 5)
train_loss, train_acc, best_eval_accuracy = train(model, 1000, 128, 0.0001, matches, device='cuda:2')
save_model(model, "predmodel_diff/final_model.pth")
print(best_eval_accuracy)

import matplotlib.pyplot as plt



plot_and_save(train_loss, train_acc, filename='diff.png')
