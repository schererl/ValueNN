import numpy as np
import copy
from typing import List, Tuple
from colorama import init, Fore
import numpy as np
import os
BOARD_SIZE = 5
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
ARROW = 3

DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, 1), (-1, -1)]  
ACTION = ['MOVE', 'THROW']

class Amazons:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_state = (PLAYER_X, 'MOVE') # ALWAYS STARTS PLAYER_X NEVER CHANGE HERE
        self.game_over = False
        self.winner = None
        self.last_moved_piece = None

        # Place amazons
        self.board[0][1] = self.board[0][3] = PLAYER_X
        self.board[4][1] = self.board[4][3] = PLAYER_O
    
        self.network_state = self._NN_state()
    def curr_mover(self):
      return self.current_state[0]
    def deep_copy(self):
        new_game = Amazons()
        new_game.board = np.copy(self.board)
        new_game.current_state = self.current_state
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.last_moved_piece = self.last_moved_piece
        return new_game
    
    def opponent(self, player):
        return PLAYER_O if player == PLAYER_X else PLAYER_X

    def next_state(self):
        if self.current_state[1] == 'MOVE':
            self.current_state = (self.current_state[0], 'THROW')
        else:
            self.current_state = (self.opponent(self.current_state[0]), 'MOVE')

    def play(self, src: Tuple[int, int], dest: Tuple[int, int], arrow: Tuple[int, int] = None):
        curr_action = self.current_state[1]
        curr_player = self.current_state[0]
        if self.game_over:
            return False
        elif curr_action == 'MOVE' and self.is_valid_move(src, dest):
            self.board[src[0]][src[1]], self.board[dest[0]][dest[1]] = EMPTY, curr_player
            self.last_moved_piece = dest  # Save the new position of the moved piece
            self.network_state = self._NN_state()
            self.next_state()
            return True
        elif curr_action == 'THROW' and self.is_valid_arrow_position(arrow):
            self.board[arrow[0]][arrow[1]] = ARROW
            self.network_state = self._NN_state()
            
            self.next_state()
            return True
        else:
            return False

    def check_game_over(self):
        curr_player = self.current_state[0]
        # Game ends when a player has no valid moves left
        has_valid_move = False
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == curr_player and self.has_valid_move((i, j)):
                    has_valid_move = True
                    break
            if has_valid_move:
                break
        if not has_valid_move:
            self.game_over = True
            self.winner = self.opponent(curr_player)
        return self.game_over

    def is_valid_move(self, src: Tuple[int, int], dest: Tuple[int, int]) -> bool:
        curr_player = self.current_state[0]
        if self.board[src[0]][src[1]] != curr_player or self.board[dest[0]][dest[1]] != EMPTY:
            return False
        dir = (np.sign(dest[0] - src[0]), np.sign(dest[1] - src[1]))
        if dir not in DIRECTIONS:
            return False
        cur = (src[0] + dir[0], src[1] + dir[1])
        while cur != dest:
            if not (0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE and self.board[cur[0]][cur[1]] == EMPTY):
                return False
            cur = (cur[0] + dir[0], cur[1] + dir[1])
        return True

    def is_valid_arrow_position(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < BOARD_SIZE and 0 <= pos[1] < BOARD_SIZE and self.board[pos[0]][pos[1]] == EMPTY

    def has_valid_move(self, src: Tuple[int, int]) -> bool:
        for dir in DIRECTIONS:
            cur = (src[0] + dir[0], src[1] + dir[1])
            while 0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE:
                if self.board[cur[0]][cur[1]] != EMPTY:
                    break
                return True
        return False
    
    
    def get_current_player_positions(self) -> List[Tuple[int, int]]:
        curr_player = self.current_state[0]
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == curr_player:
                    positions.append((i, j))
        return positions

    def get_move_positions(self, src: Tuple[int, int]) -> List[Tuple[int, int]]:
      moves = []
      for dir in DIRECTIONS:
        cur = (src[0] + dir[0], src[1] + dir[1])
        while 0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE and self.board[cur[0]][cur[1]] == EMPTY:
            moves.append(cur)
            cur = (cur[0] + dir[0], cur[1] + dir[1])
      return moves

    def get_arrow_positions(self, src: Tuple[int, int]) -> List[Tuple[int, int]]:
      arrows = []
      for dir in DIRECTIONS:
        cur = (src[0] + dir[0], src[1] + dir[1])
        while 0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE:
            if self.board[cur[0]][cur[1]] != EMPTY:
                break
            arrows.append(cur)
            cur = (cur[0] + dir[0], cur[1] + dir[1])
      return arrows



    def available_moves(self):
      moves = []
      curr_action = self.current_state[1]

      if curr_action == 'MOVE':
        for src in self.get_current_player_positions():
            for dest in self.get_move_positions(src):
                moves.append((src, dest, None))  # No arrow position needed for 'MOVE' action
      else:  # curr_action == 'THROW'
        if self.last_moved_piece is not None:
            for arrow in self.get_arrow_positions(self.last_moved_piece):
                moves.append((self.last_moved_piece, None, arrow))  # The destination remains the same for 'THROW' action

      return moves


    def _NN_state(self):
        nn_player = [1] if self.curr_mover()==PLAYER_X else [0]
        p1 = np.zeros(25)
        p2 = np.zeros(25)
        arrow = np.zeros(25)
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == PLAYER_X:
                    p1[i * len(self.board[0]) + j] = 1
                elif self.board[i][j] == PLAYER_O:
                    p2[i * len(self.board[0]) + j] = 1
                elif self.board[i][j] == ARROW:
                    arrow[i * len(self.board[0]) + j] = 1
        nn_state = np.concatenate((nn_player, p1, p2, arrow))
        return nn_state

    def print_board(self):
      for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if self.board[i][j] == PLAYER_X:
                print(Fore.RED + " X ", end="")
            elif self.board[i][j] == PLAYER_O:
                print(Fore.GREEN + " O ", end="")
            elif self.board[i][j] == ARROW:
                print(Fore.YELLOW + " ^ ", end="")
            else:
                print(Fore.WHITE + " o ", end="")
        print(Fore.WHITE) 

    def print_NN(self, name, lst):
        #print(name, end="")
        for i in lst:
            if i == 1:
                print(Fore.RED + " 1 ", end="")
            else:
                print(Fore.WHITE + " 0 ", end="")
        print(Fore.WHITE)
    
