import os
import numpy as np
import random
import time
import Amazons
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from MCTS import UCT

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import concurrent.futures
import json


class METAMODEL:
    def __init__(self, need_train=False, model_name = ''):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not need_train and os.path.isfile(model_name+'.pt'):
            self._model = None
            self.simpleNN()  # Initialize the model architecture first
            self._model.load_state_dict(torch.load(model_name+'.pt'))  # Correct way to load the state dict
            self._model.to(self._device)
            self._need_train = False
            self.trained_model = NNModel(self)
        else:
            self._need_train = True
            self._model = None
            self.trained_model = None
    
    def train(self, p1, p2, playouts, incremental=False, num_threads=12, output='empty'):
        X = []
        y = []
        progress = 0
        sum_p1_wins = 0
        lock = Lock()
        N = playouts // num_threads
        
        def thread_func():
            nonlocal progress
            nonlocal sum_p1_wins
            for i in range(N):
                local_X = []
                local_y = []
                
                hist = None
                outcome=None
                hist, outcome = self.battle(p1, p2)
                for h in reversed(hist):
                    #if(random.random() < 0.9):
                    #    continue
                    
                    local_X.append(h)
                    
                    # if player_x is the mover then the outcome is normal
                    if h[0] == 1:
                        local_y.append(outcome)
                    # if player y is the mover, the outcome is the complement
                    else:
                        local_y.append(1-outcome)
                with lock:
                    X.extend(local_X)
                    y.extend(local_y)
                    sum_p1_wins += outcome
                    progress += 1
                    #os.system('clear')
                    #print(f'progress: {sum_p1_wins}/{progress}/{playouts} ==> {output}')

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_func) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        X = np.array(X)
        y = np.array(y)
        
        if not incremental:
            self.simpleNN()
            #self.convNN()

        elif self._model is None:
            print("No model to increment. Train a model first.")
            return

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters())
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                labels = labels.view(-1, 1)
                
                optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1} - Loss: {running_loss / 200}')
                    running_loss = 0.0

            #avg_epoch_loss = running_loss / i #len(dataloader)
            #print(f'Epoch {epoch + 1} completed, average loss: {avg_epoch_loss}')

        self.trained_model = NNModel(self)

    def simpleNN(self):
        self._model = nn.Sequential(
            nn.Linear(76, 76),
            nn.ReLU(),
            nn.BatchNorm1d(76),
            nn.Dropout(0.2),
            nn.Linear(76, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.2),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.BatchNorm1d(25),
            nn.Dropout(0.2),
            nn.Linear(25, 1)
        ).to(self._device)

    def test_model(self, opponent, playouts, p2=False, str_conc=''):
        modelscore=0
        for i in range(playouts):
            if p2:
                hist, outcome = self.battle(opponent, self.trained_model)
                if outcome == 0:
                    modelscore+=1
            else:
                hist, outcome = self.battle(self.trained_model, opponent)
                if outcome == 1: 
                    modelscore+=1
            
            
            os.system('clear')
            print(f'{str_conc}\nprogress: {modelscore}/{i+1}/{playouts}') 
        return str(modelscore)+'/'+str(playouts)

    def battle(self, agent1, agent2, debug=False):
        game = Amazons.Amazons()
        s = []
        if debug:
            print(game.print_board())
            print(game.NN_state())
            print("\n")
            time.sleep(3)

        while not game.check_game_over():
            if game.curr_mover() == Amazons.PLAYER_X:
                move = agent1.evaluate(game.deep_copy())
                game.play(*move)
            else:
                move = agent2.evaluate(game.deep_copy())
                game.play(*move)
            s.append(game.NN_state())

            if debug:
                print(game.print_board())
                print(game.NN_state())
                print("\n")
                time.sleep(3)
            
        
        o = 0
        if game.winner == Amazons.PLAYER_X:
            o=1

        if debug:
            print(f'X WINNER? {o}')
            time.sleep(3)
        return s, o
    
    def store_model(self, model_name):
        torch.save(self._model.state_dict(), model_name+'.pt')
    
class AGENT:
    @abstractmethod
    def evaluate(self, game):
        pass

class ZEROMODEL(AGENT):
    def evaluate(self, game):
        return random.choice(game.available_moves())

class NNModel(AGENT):
    def __init__(self, metamodel):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = metamodel._model.to(self.device)
        
    def evaluate(self, game, debug=False):
        moves = game.available_moves()
        max_move = None
        max_value = -10000
        
        for m in moves:
            game_cpy = game.deep_copy()
            game_cpy.play(*m)
            state = torch.tensor(game_cpy.NN_state(), dtype=torch.float32).unsqueeze(0).to(self.device)
            self.model.eval()
            value = self.model(state)
            value = value.detach().item()
            if value > max_value:
                max_value = value
                max_move = m
            
            if debug:
                os.system('clear')
                game.print_board()
                print('\n')
                game_cpy.print_board()
                print(f'nn preditct: {value}')
                print(f'max {max_value} {max_move}')
                time.sleep(3)
            
            

        if debug:
            print(f'value:{max_value}')
        return max_move


class HumanAgent(AGENT):
    def evaluate(self, game:Amazons):
        moves = game.available_moves()
        choosen_move = None

        for i in range(len(moves)):
            print(i, " - ", moves[i])

        choosen_move = int(input("Select Move: "))

        return moves[choosen_move]

class STOCHASTICNN(AGENT):
    def __init__(self, metamodel, prob):
        self.model = metamodel.trained_model
        self.rnd = ZEROMODEL()
        self.prob = prob
        self.i=0
        self.I=0
    def evaluate(self, game):
        #print(f'last {self.i}/{self.I}')
        self.I+=1
        if(random.random() < self.prob):
            return self.rnd.evaluate(game)
        else:
            self.i+=1
            return self.model.evaluate(game)

class NOMODELUCT(AGENT):
    def __init__(self, model=None):
        self._model = model
    def evaluate(self, game):
        return UCT(game, 500, model=self._model, verbose=False)