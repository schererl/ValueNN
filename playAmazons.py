import Amazons
from agentes import StochasticModel, ValueNetwork, HumanAgent
from metamodel import METAMODEL
from colorama import Fore

import os
import time

'''
    Play model vs human
'''

agent = StochasticModel(ValueNetwork(METAMODEL("TOPMODEL")), 0.3)

def battle(agent1, agent2, debug=False):
    game = Amazons.Amazons()
    states = []

    game.print_board()

    time.sleep(1)

    while not game.check_game_over():

        if game.curr_mover() == Amazons.PLAYER_X:
            move = agent1.evaluate(game.deep_copy())
            game.play(*move)
        else:
            move = agent2.evaluate(game.deep_copy())
            game.play(*move)
        
        states.append(game.network_state)
        os.system('clear')
        game.print_board()
        
    if game.winner == Amazons.PLAYER_X:
        print("VENCEDOR:" + Fore.RED + "X")
    else:
        print("VENCEDOR:" + Fore.GREEN + "O")

battle(HumanAgent(), agent)