import time
import random

import numpy as np

from player import Player as P


rows, cols = 8, 8

class LazyBoi(P):

    def setup(self):
        time.sleep(1.)
        print(self.__class__.__name__ + ': ...Hmm?')


    def play(self, board: np.ndarray) -> int:
        time.sleep(random.random() + 0.2)
        return (0, 0)



#  We dont need to inherit from `player.Player` as long as the two methods are
# implemented.
class SeeWhatSticks:

    def setup(self):
        print(self.__class__.__name__ + ': Bruh I "See What Sticks", I don\'t need to prepare -_-')


    def play(self, board: np.ndarray) -> int:
        return (random.randint(0, board.shape[0]), random.randint(0, board.shape[1]))



# This is the default player. If the player class is not specified in the script
# argument, the class named `Player` is imported.
class Player(P):


    def setup(self):
        print(self.__class__.__name__ + ': The default imported class name. Pretty boring.')


    def play(self, board: np.ndarray):
        return (0, 0)



class SmartRandom(P):


    def setup(self):
        self.rows = rows
        self.columns = cols
        print('Smartttttt RRNDDDD')
        print(self.__class__.__name__ + ': The default imported class name. Pretty smart random.')


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
        print(moves)
        if len(moves) == 0:
            return None 
        else:
            a = np.random.randint(0, len(moves) )

        print(moves[a])

        return moves[a]



if __name__=='__main__':
    from othello import OthelloBoard

    print('Playing 2 dummy players against each other')
    game = OthelloBoard(rows, cols, 1, 1)


    p1 = 'arena/ted'
    p2 = 'arena/clayton'

   
    winner, reason, moves = game.play(p1, p2)
    print('Winner: %s' % winner)
    print('Reason: %s' % reason)
    print(game)