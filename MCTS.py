import numpy as np
import time
from math import sqrt, log
import random
import Amazons
#implementing
class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.available_moves = game_state.available_moves()
        self.wins = [0,0]
        self.visits = 1

    def UCT_select_child(self):
        s = sorted(self.children, key=lambda c: c.wins[self.game_state.curr_mover()-1] / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, n):
        self.children.append(n)

    def update(self, result):
        self.visits += 1
        self.wins[0] += result[0]
        self.wins[1] += result[1]

    def fully_expanded(self):
        return len(self.available_moves) == 0

def UCT(game_state, itermax, model = None, verbose=False):
    rootnode = Node(game_state)

    for i in range(itermax):
        node = rootnode
        
        # Select
        while node.fully_expanded() and not node.game_state.check_game_over():
            node = node.UCT_select_child()
            

        # Expand
        if not node.game_state.check_game_over():
            move = node.available_moves.pop(0)
            state_cpy = node.game_state.deep_copy()
            state_cpy.play(*move)
            child_node = Node(game_state, parent=node, move=move)
            node.add_child(child_node)
            node = child_node

        simul_state = child_node.game_state.deep_copy()

        rw = [0,0]
        if model != None:
            state = np.array(simul_state.NN_state()).reshape(1, -1)
            value = model.predict(state, verbose=0)
            
            if simul_state.curr_mover() == Amazons.PLAYER_X:
                rw[0] = value
                rw[1] = 1-value
            else:
                rw[0] = 1-value
                rw[1] = value
            
        # Simulate
        else:
            while not simul_state.check_game_over():
                simul_state.play(*random.choice(simul_state.available_moves()))
            rw[simul_state.winner-1] = 1

        # Backpropagate
        while node is not None:
            node.update(rw)
            node = node.parent

    return sorted(rootnode.children, key=lambda c: c.visits)[-1].move

