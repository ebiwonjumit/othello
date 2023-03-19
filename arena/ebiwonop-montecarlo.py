
import numpy as np
import math
import random

from collections import defaultdict
import time
from player import Player as P

rows, cols = 8, 8

TIMEOUT_MOVE = 1

# class MonteCarloTreeSearchNode():
#     def __init__(self, board, parent=None, parent_move=None):
#         self.board = board
#         self.parent = parent
#         self.parent_move = parent_move
#         self.children = []
#         self._number_of_visits = 0
#         self._results = defaultdict(int)
#         self._results[1] = 0
#         self._results[-1] = 0
#         self._untried_moves = None
#         self._untried_moves = self.untried_moves()
#         self.rows = 8
#         self.columns = 8
#         return
    
#     def untried_moves(self, board):
#         self._untried_moves = self.valid_moves(board)
#         return self._untried_moves
    
#     def q(self):
#         wins = self._results[1]
#         loses = self._results[-1]
#         return wins - loses
    
#     def n_visits(self):
#         return self._number_of_visits
    
#     def expand(self, board):
#      current_move = self._untried_moves.pop()
#      is_valid, next_move = self.process_move(current_move, board.copy())
#      child_node = MonteCarloTreeSearchNode(next_move, parent=self, parent_move=current_move)
#      self.children.append(child_node)
#      return child_node 

#     def is_terminal_node(self, board):
#         return self.is_game_over(board)
    
#     def rollout(self, board):
#         current_rollout_board = board.copy()
#         while not self.is_game_over(current_rollout_board):
#             possible_moves = self.valid_moves(current_rollout_board)
#             action = self.rollout_policy(possible_moves)
#             is_valid, current_rollout_board_end = self.process_move(action, current_rollout_board)
#         return self.get_final_score(current_rollout_board_end)
    
#     def backpropagate(self, result):
#         self._number_of_visits += 1.
#         self._results[result] += 1.
#         if self.parent:
#             self.parent.backpropagate(result)
    
#     def is_fully_expanded(self):
#         return len(self._untried_moves) == 0
    
#     def best_child(self, c_param=0.1):
#         choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n_visits()) / c.n())) for c in self.children]
#         return self.children[np.argmax(choices_weights)]

#     def rollout_policy(self, possible_moves):
#         return possible_moves[np.random.randint(len(possible_moves))]

#     def _tree_policy(self):
#         current_node = self
#         while not current_node.is_terminal_node():
#             if not current_node.is_fully_expanded():
#                 return current_node.expand()
#             else:
#                 current_node = current_node.best_child()
#         return current_node
    
#     def best_action(self):
#         simulation_no = 100
#         print("ENTERED INTO FUNC")
#         for i in range(simulation_no):
#             print("ENETERD IN FOR LOOP")
#             v = self._tree_policy()
#             reward = v.rollout()
#             v.backpropagate(reward)
#         return self.best_child(c_param=0.)
    

#     def valid_moves(self, board):
#         moves = []
#         for r_cnt in range(self.rows):
#             for c_cnt in range(self.columns):
#                 is_valid, _ = self.process_move((r_cnt, c_cnt), board.copy())
#                 if is_valid == True:
#                     moves.append((r_cnt, c_cnt))
        
#         return moves 

        
#     def check_row(self, vec, c):
#         c_num = vec.size
#         flanks = []
#         if (vec == 1).sum() > 0:
#             idxs = np.nonzero(vec==1)[0]
#             dists = c - idxs 
#             if c < c_num-1 and (dists<0).sum()>0:
#                 min_dist_right = np.abs(dists[dists<0]).min()
#                 if np.all(vec[c+1:c+min_dist_right]==-1) and min_dist_right>1:
#                     flanks.append(('r', min_dist_right-1))

#             if c > 0 and (dists>0).sum()>0:
#                 min_dist_left = np.abs(dists[dists>0]).min()
#                 if np.all(vec[c-min_dist_left+1:c]==-1) and min_dist_left>1:
#                     flanks.append(('l', min_dist_left-1))

#         return flanks 

#     def process_move(self, move, board):
#         r, c = move
#         if board[r, c] != 0:
#             return False, board 

#         flanks = self.check_row(board[r, :], c)
#         moved = False
#         if len(flanks) > 0:
#             moved = True
#             for dir, dist in flanks:
#                 if dir == 'r':
#                     board[r, c+1:c+dist+1] = 1

#                 if dir == 'l':
#                     board[r, c-dist:c] = 1

#         flanks = self.check_row(board[:, c].T, r)
#         if len(flanks) > 0:
#             moved = True
#             for dir, dist in flanks:
#                 if dir == 'r':
#                     board[r+1:r+1+dist, c] = 1

#                 if dir == 'l':
#                     board[r-dist:r, c] = 1

#         k = c-r
#         main_diag = np.diag(board, k)
#         flanks = self.check_row(main_diag, min(r , c))
#         if len(flanks) > 0:
#             moved = True
#             for dir, dist in flanks:
#                 if dir == 'r':
#                     tmp = board[r+1:r+dist+1, c+1:c+dist+1].copy()
#                     np.fill_diagonal(tmp, 1)
#                     board[r+1:r+dist+1, c+1:c+dist+1] = tmp

#                 if dir == 'l':
#                     tmp = board[r-dist:r, c-dist:c].copy()
#                     np.fill_diagonal(tmp, 1)
#                     board[r-dist:r, c-dist:c] = tmp

#         board = np.fliplr(board)
#         k = board.shape[1]-1-c-r 
#         side_diag = np.diag(board, k)
#         flanks = self.check_row(side_diag, min(r , board.shape[1]-1-c))
#         if len(flanks) > 0:
#             moved = True
#             for dir, dist in flanks:
#                 if dir == 'r':
#                     tmp = board[r+1:r+dist+1, c+1:c+dist+1].copy()
#                     np.fill_diagonal(tmp, 1)
#                     board[r+1:r+dist+1, c+1:c+dist+1] = tmp

#                 if dir == 'l':
#                     tmp = board[r-dist:r, c-dist:c].copy()
#                     np.fill_diagonal(tmp, 1)
#                     board[r-dist:r, c-dist:c] = tmp

#         board = np.fliplr(board)
#         if moved == True:
#             board[r, c] = 1

#         return moved, board
    
#     # Determine final score
#     def get_final_score(self, board):
#         final_score = 0
#         opponent_score = 0
#         player_score = 0
#         for r_cnt in range(self.rows):
#             for c_cnt in range(self.columns):
#                 val = board[r_cnt,c_cnt]
#                 if val == 1: player_score += 1
#                 elif val == -1: opponent_score -= 1

#         if player_score > opponent_score: final_score += 1
#         elif opponent_score > player_score: final_score -=1
#         return final_score
    
    
#     # Check if there are moves left
#     def is_game_over(self, board):
#         moves = self.valid_moves(board)
#         if len(moves) == 0:
#             return True
#         return False
# class MCTSNode():
#     def __init__(self, state, parent=None):
#         self.state = state
#         self.parent = parent
#         self.children = []
#         self.visits = 0
#         self.total_reward = 0
    
#     def add_child(self, child_state):
#         child = MCTSNode(child_state, self)
#         self.children.append(child)
#         return child
    
#     def update(self, reward):
#         self.visits += 1
#         self.total_reward += reward
    
#     def fully_expanded(self):
#         return len(self.children) == len(self.state.get_valid_actions())
    
#     def best_child(self, c_param=1.4):
#         best_score = -float("-inf")
#         best_child = None
#         for child in self.children:
#             exploitation = child.total_reward / child.visits
#             exploration = math.sqrt(math.log(self.visits) / child.visits))
#             score = exploitation + c_param * exploration
#             if score > best_score:
#                 best_score = score
#                 best_child = child
#         return best_child



class Player():

    def setup(self):
        self.rows = rows
        self.columns = cols


    def valid_moves(self, board):
        moves = []
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                is_valid, _ = self.process_move((r_cnt, c_cnt), board.copy())
                if is_valid == True:
                    moves.append((r_cnt, c_cnt))
        
        return moves 

        
    def check_row(self, vec, c):
        c_num = vec.size
        flanks = []
        if (vec == 1).sum() > 0:
            idxs = np.nonzero(vec==1)[0]
            dists = c - idxs 
            if c < c_num-1 and (dists<0).sum()>0:
                min_dist_right = np.abs(dists[dists<0]).min()
                if np.all(vec[c+1:c+min_dist_right]==-1) and min_dist_right>1:
                    flanks.append(('r', min_dist_right-1))

            if c > 0 and (dists>0).sum()>0:
                min_dist_left = np.abs(dists[dists>0]).min()
                if np.all(vec[c-min_dist_left+1:c]==-1) and min_dist_left>1:
                    flanks.append(('l', min_dist_left-1))

        return flanks 

    def process_move(self, move, board):
        r, c = move
        if board[r, c] != 0:
            return False, board 

        flanks = self.check_row(board[r, :], c)
        moved = False
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    board[r, c+1:c+dist+1] = 1

                if dir == 'l':
                    board[r, c-dist:c] = 1

        flanks = self.check_row(board[:, c].T, r)
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    board[r+1:r+1+dist, c] = 1

                if dir == 'l':
                    board[r-dist:r, c] = 1

        k = c-r
        main_diag = np.diag(board, k)
        flanks = self.check_row(main_diag, min(r , c))
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r+1:r+dist+1, c+1:c+dist+1].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r+1:r+dist+1, c+1:c+dist+1] = tmp

                if dir == 'l':
                    tmp = board[r-dist:r, c-dist:c].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r-dist:r, c-dist:c] = tmp

        board = np.fliplr(board)
        k = board.shape[1]-1-c-r 
        side_diag = np.diag(board, k)
        flanks = self.check_row(side_diag, min(r , board.shape[1]-1-c))
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r+1:r+dist+1, c+1:c+dist+1].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r+1:r+dist+1, c+1:c+dist+1] = tmp

                if dir == 'l':
                    tmp = board[r-dist:r, c-dist:c].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r-dist:r, c-dist:c] = tmp

        board = np.fliplr(board)
        if moved == True:
            board[r, c] = 1

        return moved, board

    def play(self, board: np.ndarray):
        moves = self.valid_moves(board)

        # print(moves)
        if len(moves) == 0:
            return None
        else:
            a = np.random.randint(0, len(moves) )
            # root = MonteCarloTreeSearchNode(board, None, None)
            # print("I STARTED A NEW CLASS")
            # selected_node = root.best_action
            # print("WE SELECTED")
            # print(selected_node)
            b = self.get_best_move(tree, board.copy(), 20)

        print('------------ RANDO -----------')    
        print(moves[a])
        print('------------ MONTE CARLO  -----------')    
        print(b)

        return moves[a]
    


    def get_best_move(self, tree, board, simulations):
        for i in range(simulations):
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child(c_param=0.)


    # Main Monte Carlo Function
    def monte_carlo(self, board):
        current_state = self.valid_moves(board) 
        best_move = None
        best_score = float("-inf") 
        return best_move


    # Rollout policy
    def rollout_policy(self,possible_moves):
        return np.random.randint(0, len(possible_moves))
    
    # # Tree Policy
    # def tree_policy(self, board):
    #     current_state = board.copy()
    #     while not self.is_game_over(current_state):
    #         if not current_state



# Check if there are moves left
    def is_game_over(self, board):
        moves = self.valid_moves(board)
        if len(moves) == 0:
            return True
        return False

# Heuristic eval function
# Subtract opponent pieces from player pieces    
    def check_score(self, board):
        score = 0
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                val = board[r_cnt,c_cnt]
                if val == 1: score += 1
                elif val == -1: score -= 1
        return score
            
