import copy
import sys

import numpy as np

from player import Player as P
rows, cols = 8, 8

class Player():


    def setup(self):
        """
        This method will be called once at the beginning of the game so the player
        can conduct any setup before the move timer begins. The setup method is
        also timed.
        """
        self.rows = rows
        self.columns = cols
        print(self.__class__.__name__ + ': Alpha Beta.')

#calculate the socres of the player
    def evaluate(self,board):
        mine = 0
        other = 0
        for row in range(8):
            for col in range(8):
                if board[row,col] == 1 : mine+=1
                elif board[row,col] == -1 : other+=1
        return mine


    def valid_moves(self, board):
        moves = []
        for r_cnt in range(8):
            for c_cnt in range(8):
                is_valid, _ = self.process_move((r_cnt, c_cnt), board.copy())
                if is_valid == True:
                    moves.append((r_cnt, c_cnt))

        return moves

    def check_row(self, vec, c):
        c_num = vec.size
        flanks = []
        if (vec == 1).sum() > 0:
            idxs = np.nonzero(vec == 1)[0]
            dists = c - idxs
            if c < c_num - 1 and (dists < 0).sum() > 0:
                min_dist_right = np.abs(dists[dists < 0]).min()
                if np.all(vec[c + 1:c + min_dist_right] == -1) and min_dist_right > 1:
                    flanks.append(('r', min_dist_right - 1))

            if c > 0 and (dists > 0).sum() > 0:
                min_dist_left = np.abs(dists[dists > 0]).min()
                if np.all(vec[c - min_dist_left + 1:c] == -1) and min_dist_left > 1:
                    flanks.append(('l', min_dist_left - 1))

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
                    board[r, c + 1:c + dist + 1] = 1

                if dir == 'l':
                    board[r, c - dist:c] = 1

        flanks = self.check_row(board[:, c].T, r)
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    board[r + 1:r + 1 + dist, c] = 1

                if dir == 'l':
                    board[r - dist:r, c] = 1

        k = c - r
        main_diag = np.diag(board, k)
        flanks = self.check_row(main_diag, min(r, c))
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r + 1:r + dist + 1, c + 1:c + dist + 1].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r + 1:r + dist + 1, c + 1:c + dist + 1] = tmp

                if dir == 'l':
                    tmp = board[r - dist:r, c - dist:c].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r - dist:r, c - dist:c] = tmp

        board = np.fliplr(board)
        k = board.shape[1] - 1 - c - r
        side_diag = np.diag(board, k)
        flanks = self.check_row(side_diag, min(r, board.shape[1] - 1 - c))
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r + 1:r + dist + 1, c + 1:c + dist + 1].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r + 1:r + dist + 1, c + 1:c + dist + 1] = tmp

                if dir == 'l':
                    tmp = board[r - dist:r, c - dist:c].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r - dist:r, c - dist:c] = tmp

        board = np.fliplr(board)
        if moved == True:
            board[r, c] = 1

        return moved, board


    def alphabeta(self, board, alpha, beta, depth, bool):

        if bool == True:
            if depth == 0:
                return self.evaluate(board)


            best_val = -float('inf')
            moves = self.valid_moves(board)
            for move in moves:

                moved,new_board = self.process_move(move,board.copy())

                val = self.alphabeta(-new_board, alpha, beta, depth - 1, False)

                # best_val = max(best_val, val)
                best_val = max(best_val, val)
                if best_val >= beta:
                    return best_val
                alpha = max(alpha,best_val)

            return best_val

        else:
            if depth == 0:
                return self.evaluate(board)
            best_val = float('inf')
            moves = self.valid_moves(board)
            for move in moves:

                moved, new_board = self.process_move(move, board.copy())

                val = self.alphabeta(-new_board, alpha, beta, depth - 1, True)

                best_val = min(best_val, val)
                if best_val <= alpha:
                    return best_val
                beta = min(beta, best_val)

            return best_val


    def play(self, board: np.ndarray):
        depth = 4
        best_move = None
        moves = self.valid_moves(board)
        if len(moves) == 0:
            return None
        best_val = -float('inf')
        for move in moves:
            _,new_board = self.process_move(move, board.copy())
            move_val = self.alphabeta(-new_board, -1000, 1000, depth-1, False)
            if move_val > best_val:
                best_move = move
                best_val = move_val

        return best_move
