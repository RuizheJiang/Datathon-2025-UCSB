import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import *

class MatchSampler(torch.utils.data.Dataset):
    def __init__(self, matches):
        self.matches = matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx][0]
        return_dic = {
            "my_cards" : match.winner_info["cards"],
            "my_features" : np.array([match.winner_info["troop_count"], match.winner_info["structure_count"], match.winner_info["spell_count"], match.winner_info["elixir_average"]]),
            "my_Trophies" : match.winner_info["Trophies"],
            "my_troop_count" : match.winner_info["troop_count"],
            "my_structure_count" : match.winner_info["structure_count"],
            "my_spell_count" : match.winner_info["spell_count"],
            "my_elixir_average" : match.winner_info["elixir_average"],

            "opponent_cards" : match.loser_info["cards"],
            "opponent_features" : np.array([match.loser_info["troop_count"], match.loser_info["structure_count"], match.loser_info["spell_count"], match.loser_info["elixir_average"]]),
            "opponent_Trophies" : match.loser_info["Trophies"],
            "opponent_troop_count" : match.loser_info["troop_count"],
            "opponent_structure_count" : match.loser_info["structure_count"],
            "opponent_spell_count" : match.loser_info["spell_count"],
            "opponent_elixir_average" : match.loser_info["elixir_average"],

            "labels" : self.matches[idx][1]
        }

        return return_dic


class PredModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(PredModel, self).__init__()

        input_dim = input_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)  # Input layer
        self.attn_layer_1 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.attn_layer_2 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, deck1, deck2, feature1, feature2):
        x = torch.cat([deck1-deck2], axis=-1)
        y = torch.relu(self.input_layer(x))
        y = y.unsqueeze(1)
        attn_output_1, _ = self.attn_layer_1(y, y, y)
        attn_output_2, _ = self.attn_layer_2(attn_output_1, attn_output_1, attn_output_1)
        attn_output_2 = attn_output_2.squeeze(1)

        residual = attn_output_2
        for layer in self.hidden_layers:
            y = torch.relu(layer(attn_output_2))
            y = y + residual
            residual = y
        win_probability = torch.sigmoid(self.output_layer(y))  # Win probability (0-1)
        return win_probability


class PredModel_withConcat(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(PredModel_withConcat, self).__init__()

        input_dim = input_dim * 2
        self.input_layer = nn.Linear(input_dim, hidden_dim)  # Input layer
        self.attn_layer_1 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.attn_layer_2 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, deck1, deck2, feature1, feature2):
        x = torch.cat([deck1, deck2], axis=-1)
        y = torch.relu(self.input_layer(x))
        y = y.unsqueeze(1)
        attn_output_1, _ = self.attn_layer_1(y, y, y)
        attn_output_2, _ = self.attn_layer_2(attn_output_1, attn_output_1, attn_output_1)
        attn_output_2 = attn_output_2.squeeze(1)

        residual = attn_output_2
        for layer in self.hidden_layers:
            y = torch.relu(layer(attn_output_2))
            y = y + residual
            residual = y
        win_probability = torch.sigmoid(self.output_layer(y))  # Win probability (0-1)
        return win_probability


class PredModel_withFeatures(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(PredModel_withFeatures, self).__init__()

        input_dim = input_dim * 2 + 8
        self.input_layer = nn.Linear(input_dim, hidden_dim)  # Input layer
        self.attn_layer_1 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.attn_layer_2 = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, deck1, deck2, feature1, feature2):
        x = torch.cat([deck1, deck2, feature1, feature2], axis=-1)
        y = torch.relu(self.input_layer(x))
        y = y.unsqueeze(1)
        attn_output_1, _ = self.attn_layer_1(y, y, y)
        attn_output_2, _ = self.attn_layer_2(attn_output_1, attn_output_1, attn_output_1)
        attn_output_2 = attn_output_2.squeeze(1)

        residual = attn_output_2
        for layer in self.hidden_layers:
            y = torch.relu(layer(attn_output_2))
            y = y + residual
            residual = y
        win_probability = torch.sigmoid(self.output_layer(y))  # Win probability (0-1)
        return win_probability
