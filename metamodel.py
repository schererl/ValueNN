#UTILS
import os
import numpy as np
import random
import time

# TORCH
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# MULTITHREAD
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import concurrent.futures

# PROJECT
import Amazons
from agentes import ValueNetwork


model_file_name = 'MODELS/'
class METAMODEL:
    def __init__(self, need_train=False, model_name = ''):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelX = self.simpleNN().to(self._device)
        self.modelO = self.simpleNN().to(self._device)
       
        if not need_train:
            self.modelX.load_state_dict(torch.load(model_file_name+model_name+'X.pt'))
            self.modelX.to(self._device)
            self.modelO.load_state_dict(torch.load(model_file_name+model_name+'O.pt'))
            self.modelO.to(self._device)
            
            
            self.agent_model = ValueNetwork(self)
        
    def train_models(self, p1, p2, playouts, num_threads=12, output='empty'):
        self._train(p1, p2, playouts/2, True,  num_threads, output)
        self._train(p1, p2, playouts/2, False, num_threads, output)
        self.agent_model = ValueNetwork(self)

    def _train(self, p1, p2, playouts, train_p1, num_threads, output):
        X = []
        y = []
        progress = 0
        sum_p1_wins = 0
        N = (int)(playouts // num_threads)
        print(playouts)
        print(num_threads)
        lock = Lock()
        def thread_func():
            nonlocal progress
            nonlocal sum_p1_wins
            for _ in range(N):
                local_X = []
                local_y = []
                hist, outcome = self.battle(p1, p2)
                h_out = outcome
                
                for h in reversed(hist):
                    #time.sleep(2)
                    if h[0] == 0 and (not train_p1): # jogada do P2
                        h_out = 1-outcome
                        local_X.append(h)
                        local_y.append(h_out)
                        #print(f'X outcome {outcome} player {h[0]} player outcome {h_out}')
                    elif h[0] == 1 and train_p1: # jogada do P1
                        local_X.append(h)
                        local_y.append(h_out)
                        #print(f'X outcome {outcome} player {h[0]} player outcome {h_out}')
                with lock:
                    X.extend(local_X)
                    y.extend(local_y)
                    sum_p1_wins += outcome
                    progress += 1
                    os.system('clear')
                    print(f'progress: {sum_p1_wins}/{progress}/{playouts} ==> {output}')
                

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_func) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
            
        if train_p1:
            self.train_NN(self.modelX, X, y)
        else:
            self.train_NN(self.modelO, X, y)
        
        

    def train_NN(self, model, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                labels = labels.view(-1, 1)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:  
                    print(f'Epoch {epoch + 1}, Batch {i + 1} - Loss: {running_loss / 200}')
                    running_loss = 0.0

    def simpleNN(self):
        return nn.Sequential(
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
        )

    def test_model(self, opponent, playouts, p2=False, str_conc=''):
        modelscore=0
        for i in range(playouts):
            if p2:
                hist, outcome = self.battle(opponent, self.agent_model)
                if outcome == 0:
                    modelscore+=1
            else:
                hist, outcome = self.battle(self.agent_model, opponent)
                if outcome == 1: 
                    modelscore+=1
            
            
            os.system('clear')
            print(f'{str_conc}\nprogress: {modelscore}/{i+1}/{playouts}') 
        return str(modelscore)+'/'+str(playouts)

    def battle(self, agent1, agent2, debug=False):
        game = Amazons.Amazons()
        states = []
        while not game.check_game_over():
            if game.curr_mover() == Amazons.PLAYER_X:
                move = agent1.evaluate(game.deep_copy())
                game.play(*move)
            else:
                move = agent2.evaluate(game.deep_copy())
                game.play(*move)
            states.append(game.network_state)
            
         
        if game.winner == Amazons.PLAYER_X:
            return states, 1
        return states, 0

        
    
    def store_model(self, model_name):
        torch.save(self.modelX.state_dict(), model_file_name+model_name+'X.pt')
        torch.save(self.modelO.state_dict(), model_file_name+model_name+'O.pt')