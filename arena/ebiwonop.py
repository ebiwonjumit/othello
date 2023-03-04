
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
        chosen_move = []
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
            chosen_move.append((r,c))
            board[r, c] = 1

        return moved, board


    def play(self, board: np.ndarray):
        moves = self.valid_moves(board)
        # print(moves)
        if len(moves) == 0:
            return None 
        else:
            a = np.random.randint(0, len(moves) )
            b = self.minimaxer(board)

        print('------------ ALPHAOTHELLO -----------')    
        print(moves[a])

        return moves[a]
    
    # def alphabeta(self, board, depth, alpha, beta, maximizePlayer):
    #     moves = self.valid_moves(board)
    #     if depth == 0 or len(moves) == 0:
            
    #         return moves[0]
        
    #     if maximizePlayer: 
    #         maxEval = float("-inf")
    #         for m in range(0, len(moves)): 
    #             is_valid, my_board = self.process_move(moves[m], board)
    #             print("WE GOT TO THE FIFTH LINE")
    #             eval = self.alphabeta(self, my_board, depth - 1, alpha, beta, False)
    #             print("I AM TRYING TO PRINT MY RESULTS")
    #             print(eval)
    #             maxEval = max(maxEval,eval)
    #             alpha = max(alpha,eval)
    #             if beta <= alpha:
    #                 break
    #         return maxEval
    #     else:
    #         minEval = float("inf")
    #         for m in range(0, len(moves)):
    #             is_valid, current_board = self.process_move(moves[m], board)
    #             opponent_board = np.multiply(current_board, -1)
    #             eval = self.alphabeta(opponent_board, depth + 1, alpha, beta, True)
    #             minEval = min(minEval, eval)
    #             beta = min(beta, eval)
    #         return minEval
        
    def minimax_max(self, board):
        moves = self.valid_moves(board)
        if len(moves) == 0:
            return self.check_score(board)
        
        value = float("-inf")
        for i in range(0, len(moves)):
            is_valid, current_board = self.process_move(moves[i],board)
            value = max(value, self.minimax_min(current_board))
        
        return value
    
    def minimax_min(self,board):
        opponent_board = np.multiply(board, -1)
        moves = self.valid_moves(opponent_board)
        if len(moves) == 0:
            return self.check_score(opponent_board)
        
        value = float("inf")
        for i in range(0, len(moves)):
            is_valid, current_board = self.process_move(moves[i],opponent_board )
            value = min(value, self.minimax_max(current_board))
        
        return value
    
    def minimaxer(self, board):
        moves = self.valid_moves(board)
        move_picked = []
        indexes = []

        for i in range(0, len(moves)):
            is_valid, current_board = self.process_move(moves[i], board)
            indexes.append(self.minimax_max(current_board))
        return moves[indexes.index(max(indexes))]

    def check_score(self, board):
        player, opponent = 0,0
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                val = board[r_cnt,c_cnt]
                if val == 1: player += 1
                elif val == -1: opponent += 1
        return player - opponent
            

