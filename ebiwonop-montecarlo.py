
import numpy as np
import math
from player import Player as P 
from montecarlo import MonteCarloTreeSearchNode

rows, cols = 8, 8
simulations = 3
exploration = 0.1


TIMEOUT_MOVE = 1


class Player():

    
    def setup(self):
        self.rows = rows
        self.columns = cols
        self.simulations = simulations
        self.exploration = exploration
        


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
            b = self.get_mcts_move(board)
            # selected_node = root.best_action
            # print("WE SELECTED")
            # print(selected_node)
            # b = self.get_best_move(tree, board.copy(), 20)

        print('------------ RANDO -----------')    
        print(moves[a])
        print('------------ MONTE CARLO  -----------')    
        print(b)

        return moves[a]
    


    def get_mcts_move(self,board):
        root_node = {'state': board.copy(), 'num_visits': 0, 'total_reward': 0}
        
        for i in range(self.simulations):
            node = root_node
            path = [node]
            while 'children' in node:
                node = self.uct_select(node)
                path.append(node)
            reward = self.mcts_simulate(node['state'])
            print("WE SELECTED")
            self.mcts_backpropagate(path,reward)
        children = root_node['children']
        best_child = max(children, key=lambda c: c['num_visits'])
        print(best_child)
        return best_child['move']



    def uct_select(self, node):
        log_total_visits = math.log(node['num_visits'])
        best_child = None
        best_score = float('-inf')
        for child in node['children']:
            exploit_score = child['total_reward'] / child['num_visits']
            explore_score = self.exploration * math.sqrt(log_total_visits / child['num_visits'])
            uct_score = exploit_score + explore_score
            if uct_score > best_score:
                best_child = child
                best_score = uct_score
        return best_child
    

    def mcts_simulate(self, board):
        while not self.is_game_over(board):
            move = np.random.randint(0, len(self.valid_moves(board)))
            is_valid, current_board = self.process_move(move, board.copy())
        print("WE SELECTED")
        return self.game_result(current_board)
    
    # MCTS Backpropagate
    def mcts_backpropagate(self, path, reward):
        for node in reversed(path):
            node['num_visits'] += 1
            node['total_reward'] += reward


# Check if there are moves left
    def is_game_over(self, board):
        moves = self.valid_moves(board)
        player_score = 0
        opponent_score = 0
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                val = board[r_cnt,c_cnt]
                if val == 1: player_score += 1
                elif val == -1: opponent_score += 1
        
        if len(moves) == 0:
            print("WE SELECTED TRUE")
            return True
        elif player_score + opponent_score == 64: 
            print("WE SELECTED TRUE")
            return True
        print("WE SELECTED FALSE")
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
    

    def game_result(self, board):
        player_score = 0
        opponent_score = 0
        final_score = 0
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                val = board[r_cnt,c_cnt]
                if val == 1: player_score += 1
                elif val == -1: opponent_score += 1

        if player_score + opponent_score == 64: return player_score - opponent_score
        else:   return None
            
