from abc import ABC, abstractmethod
import random
import time
import os

import torch
import torch.nn as nn

import Amazons
class AGENT:
    @abstractmethod
    def evaluate(self, game):
        pass

class NoModel(AGENT):
    def evaluate(self, game):
        return random.choice(game.available_moves())

class ValueNetwork(AGENT):
    def __init__(self, metamodel):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelX = metamodel.modelX.to(self.device)
        self.modelO = metamodel.modelO.to(self.device)
        
    def evaluate(self, game, debug=False):
        current_mover = game.curr_mover()
        moves = game.available_moves()
        max_move = None
        max_value = -10000
        
        for m in moves:
            game_cpy = game.deep_copy()
            game_cpy.play(*m)
            state = torch.tensor(game_cpy.network_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            value = 0
            if current_mover == Amazons.PLAYER_X:
                self.modelX.eval()
                value = self.modelX(state)
                value = value.detach().item()
            else:
                self.modelO.eval()
                value = self.modelO(state)
                value = value.detach().item()
                
            if value > max_value:
                max_value = value
                max_move = m
            
            if debug:
                os.system('clear')
                game.print_board()
                print('\n')
                game_cpy.print_board()
                print(f'nn preditct {value}')
                print(f'max {max_value} move {max_move}')
                print(f'mover {current_mover}\n')
                print(game_cpy.network_state)
                time.sleep(3)
        
        if debug:
            print(f'value:{max_value}')
        return max_move

class StochasticModel(AGENT):
    def __init__(self, agent, prob):
        self.agent = agent
        self.rnd = NoModel()
        self.prob = prob
    def evaluate(self, game):
        #print(f'last {self.i}/{self.I}')
        if(random.random() < self.prob):
            return self.rnd.evaluate(game)
        else:
            return self.agent.evaluate(game)
        

class HumanAgent(AGENT):
    def evaluate(self, game:Amazons):
        moves = game.available_moves()
        choosen_move = None

        os.system('clear')

        game.print_moves_board(moves)

        for i in range(len(moves)):
            print(i, " - ", moves[i])

        choosen_move = int(input("Select Move Number: "))

        return moves[choosen_move]