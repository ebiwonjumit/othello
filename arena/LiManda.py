import numpy as np
from player import Player as P
import random

import copy
from copy import deepcopy

rows, cols = 8, 8

class State:
    def __init__(self, board):
        self.board = board

    def get_board(self):
        return self.board

class Node():
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.visits = 0
        self.score = 0.0
        self.children = []
        self.children_map = {}

    def get_board(self):
        return self.state.get_board()

    def is_fully_expanded(self):
        #print("Number of valid move:",len(self.state.valid_moves(self.state)))
        #print("Number of child node:",len(self.children))
        return len(self.children) == len(self.state.valid_moves(self.state))
    
    def add_child(self, state):
        node = Node(self, state)
        self.children.append(node)
        self.children_map[state.last_move] = node
        return node

    def get_best_child(self, exploration):
        best_score = -1
        best_child = None
        print("best child in function",self.children)
        for child in self.children:
            score = child.score / child.visits + exploration * np.sqrt(2 * np.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def update(self, score):
        self.visits += 1
        self.score += score

class MCTSAgent(P):
    def __init__(self, num_simulations=10, exploration=1.0,current_player=-1):
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.current_player=current_player
        self.board=[]

    def setup(self):
        self.rows = rows
        self.columns = cols
        print('MCTS Agent')
        print(self.__class__.__name__ + ': The default imported class name. MCTS Agent.')

    def valid_moves(self, board):
        moves = []
        #print(type(board))
        #print(type(board.board))
        #print(board.board)
        if isinstance(board, np.ndarray):
            newboard=board
        else:newboard=board.board
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                is_valid, _ = self.process_move((r_cnt, c_cnt), newboard.copy())
                if is_valid == True:
                    moves.append((r_cnt, c_cnt))
        #print(moves)
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

    def last_move(self):
        # Find the last non-empty cell in the board
        for row in range(8):
            for col in range(8):
                if self.board[row][col] != 0:
                    return (row, col)
        # If the board is empty, return None
        return None
    
    def process_move(self, move, board):
        r, c = move
        if board[r, c] != 0 and board[r,c] != -0:
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

        #print(moved,board)
        return moved, board


    def play(self, board: np.ndarray):
        print(board,type(board))
        #root_state=State(board)
        self.board=board
        #print(type(root_state))
        root = Node(board,self)
        best_move=[]
        for i in range(self.num_simulations):
            node = root
            state = self.copy()
            # Select
            print(node.is_fully_expanded())
            print("Node vistited:",node.visits)
            while node.is_fully_expanded() and node.children:
                node = node.get_best_child(self.exploration)
                state.process_move(node.state, board)
            # Expand
            if not node.is_fully_expanded() or node.visits>0:
                moves = state.valid_moves(board)
                print("Valid moves:",moves)
                random.shuffle(moves)
                for move in moves:
                    if move not in [child.state for child in node.children]:
                        print("Move steps:",move)
                        new_state = state.copy()
                        new_state.process_move(move, board)
                        child_node = node.add_child(new_state)
                        print("Child before simulate:",type(child_node))
                        # Simulate
                        score = self.simulate(child_node.state, board)#score =-1or 1
                        print("board after simulation:",board)
                        print("Score of child:",score)
                        # Backpropagate
                        #print(type(child_node))
                        while child_node is not None and isinstance(child_node, Node):
                            child_node.update(score)
                            child_node = child_node.parent
                        break
        best_child = root.get_best_child(0)
        print("best_children state:",best_child)
        print(dir(best_child.state))
        best_move = move
        #for move, child in best_child.children_map.items():
        #    if child.visits == best_child.visits:
        #        best_move = move
        #        break
        return best_move
        #return best_child.state

    def simulate(self, state, board):
        moves = state.valid_moves(board)#return a list of moves
        #print(type(state.current_player))
        if not moves:
            return -1 if state.current_player == 1 else 1 # Simulate until return -1 or 1
        else:
            random_move = random.choice(moves)
            state.process_move(random_move, board)
            return self.simulate(state, board)

    def copy(self):
        new_player = MCTSAgent(self.num_simulations, self.exploration)
        new_player.rows = self.rows
        new_player.columns = self.columns
        return new_player
