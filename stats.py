from Amazons import Amazons
from agentes import StochasticModel, NoModel, ValueNetwork
from metamodel import METAMODEL
import os
import json
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
_stats_folder = 'STATISTICS/'
def store_states(model, n_games, filename):
    states = []

    for _ in range(n_games):
        game = Amazons()
        while not game.check_game_over():
            move = model.evaluate(game.deep_copy())
            game.play(*move)
            states.append(game.network_state.tolist())  # Convert ndarray to list here

    with open(_stats_folder+filename, 'w') as f:
        json.dump(states, f)


def compute_probabilities(filename):
    with open(_stats_folder+filename, 'r') as f:
        states = json.load(f)
    
    total_moves = len(states)
    ones_count = [0]*len(states[0])

    for state in states:
        for i, val in enumerate(state):
            if val == 1:
                ones_count[i] += 1

    probabilities = [count/total_moves for count in ones_count]

    return probabilities

def print_probabilities(lst):
    mover = lst[0:1]
    p1 = lst[1:26]
    p2 = lst[26:51]
    arrow = lst[51:]

    mover = ["{:.2f}".format(i) for i in mover]
    p1 = ["{:.2f}".format(i) for i in p1]
    p2 = ["{:.2f}".format(i) for i in p2]
    arrow = ["{:.2f}".format(i) for i in arrow]

    print(f"Mover {len(mover)}: {mover}")
    print(f"P1 {len(p1)}: {p1}")
    print(f"P2 {len(p2)}: {p2}")
    print(f"Arrow {len(arrow)}: {arrow}")

def generate_heatmaps():
    # Compute the probabilities
    random_agent_probs = compute_probabilities('ST-rnd')
    model_agent_probs = compute_probabilities('ST-topmodel')

    # Split your probabilities into the categories you're interested in
    random_agent_p1 = random_agent_probs[1:26]
    random_agent_p2 = random_agent_probs[26:51]
    random_agent_arrow = random_agent_probs[51:]

    model_agent_p1 = model_agent_probs[1:26]
    model_agent_p2 = model_agent_probs[26:51]
    model_agent_arrow = model_agent_probs[51:]

    # Create DataFrames for each category
    data_p1 = {
        'R P1': random_agent_p1,
        'M P1': model_agent_p1
    }
    data_p2 = {
        'R P2': random_agent_p2,
        'M P2': model_agent_p2
    }
    data_arrow = {
        'R Arrow': random_agent_arrow,
        'M Arrow': model_agent_arrow
    }

    df_p1 = pd.DataFrame(data_p1)
    df_p2 = pd.DataFrame(data_p2)
    df_arrow = pd.DataFrame(data_arrow)

    # Plot the heatmaps
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)  # 1 row, 3 cols, subplot 1
    sns.heatmap(df_p1, annot=True, cmap='viridis_r', linewidths=0.5)
    plt.title('P1 State Position Heatmap')
    plt.xlabel('Agents')
    plt.ylabel('Board Positions')

    plt.subplot(1, 3, 2)  # 1 row, 3 cols, subplot 2
    sns.heatmap(df_p2, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('P2 State Position Heatmap')
    plt.xlabel('Agents')

    plt.subplot(1, 3, 3)  # 1 row, 3 cols, subplot 3
    sns.heatmap(df_arrow, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('Arrow State Position Heatmap')
    plt.xlabel('Agents')

    plt.tight_layout()  # To ensure proper spacing between subplots
    plt.show()