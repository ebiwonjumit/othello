
import numpy as np
from player import Player as P

rows, cols = 8, 8

TIMEOUT_MOVE = 1


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
            #Calling Alpha beta with depth
            b = self.get_best_move(board, 1)

        print('------------ RANDO -----------')    
        print(moves[a])
        print("------ALPHAB MOVE")
        print(b)

        return b
    

# Helper function for running Alpha Beta
    def get_best_move(self, board, depth):
        moves = self.valid_moves(board)
        best_move = None
        best_score = float("-inf")
        for m in range(0, len(moves)):
                is_valid, potential_board = self.process_move(moves[m], board.copy())
                if is_valid == True:
                    score = self.alphabeta(potential_board, depth, float("-inf"), float("inf"), True)
                    print("HERE IS A SCORE FRIEND")
                    print(score)
                    if score > best_score:
                        best_score = score
                        best_move = moves[m]
        print(best_move)
        return best_move
    
    def alphabeta(self, board, depth, alpha, beta, maximizePlayer):
        moves = self.valid_moves(board)
        if depth == 0 or len(moves) == 0:
            return self.check_score(board)
        
        if maximizePlayer: 
            maxEval = float("-inf")
            for m in range(0, len(moves)): 
                is_valid, current_board = self.process_move(moves[m], board.copy())
                # print("HERE IS MY BOARD")
                # print(current_board)
                if is_valid == True:
                    eval = self.alphabeta(current_board, (depth - 1), alpha, beta, False)
                    maxEval = max(maxEval,eval)
                    alpha = max(alpha,eval)
                    if beta <= alpha:
                        # print("IVE CUT")
                        break
            print("HERE IS THE EVAL")
            print(maxEval)
            return maxEval
        else:
            minEval = float("inf")
            for m in range(0, len(moves)):
                is_valid, current_board = self.process_move(moves[m], board.copy())
                # print("I AM TRYING TO PRINT MY RESULTS")
                if is_valid == True:
                    opponent_board = np.multiply(current_board, -1)
                    eval = self.alphabeta(opponent_board, (depth + 1), alpha, beta, True)
                    minEval = min(minEval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            print("HERE IS THE minEVAL")
            print(minEval)
            return minEval
        


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
            

