import importlib
import random
from argparse import ArgumentParser
import sys
import os
from itertools import combinations
from typing import Dict, Iterable, Tuple
import time 
from contextlib import contextmanager
import threading
import _thread

import numpy as np

from player import Player


# board is a numpy array
# empty spots are 0s
# Internally:
#   player 1 pieces are +1
#   player 2 pieces are -1
# each player sees their pieces as +1, and their opponent as -1
# timed out action implies opponent wins

ROWS = 8
COLUMNS = 8

TIMEOUT_MOVE = 1
TIMEOUT_SETUP = 2
MAX_INVALID_MOVES = 0
timed_out = False 




class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    global timed_out

    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    
    try:
        yield
    except KeyboardInterrupt:
        # raise TimeoutException("Timed out for operation {}".format(msg))
        timed_out = True
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


class OthelloBoard():

    def __init__(
        self, rows=ROWS, columns=COLUMNS, timeout_move=TIMEOUT_MOVE,
        timeout_setup=TIMEOUT_SETUP, max_invalid_moves=MAX_INVALID_MOVES, 
        deterministic: bool=True,
        suppress: bool=False
    ):
        """
        rows : int -- number of rows in the game
        columns : int -- number of columns in the game
        time_out_secs : float -- time in seconds after which other player is declared winner
        """

        # collect max and min player vunetids for logging scores
        self.agents = {}
        self.rows = rows
        self.columns = columns
        self.timeout_move = timeout_move
        self.timeout_setup = timeout_setup
        self.max_invalid_moves = max_invalid_moves
        self.deterministic = deterministic
        self.total_time_limit = 45.0 # 45 seconds
        self.suppress = suppress
        self.time_p1 = 0
        self.time_p2 = 0

        self.reset_board()


    def __str__(self) -> str:
        board = self._board.copy().astype(int)
        return np.array2string(board)
    


    def load_players(self, p_path):
        class_name = 'Player'
        print(p_path)
        try:
            components = p_path.split('/')
            
            module_name = components[0]
            player_module = importlib.import_module(module_name)

            if len(components) == 2:
                class_name = components[1]

            # elif len(components) == 3:
            #     path_name = components[0]
            #     sys.path.append(f"./{path_name}")
            #     module_name = components[1]
            #     player_module = importlib.import_module( module_name)
            #     class_name = components[2]

        except Exception as exc:
            print('Could not load player %s due to: %s' % (p_path, exc))
            return -1
        player_cls = getattr(player_module, class_name)
        player: Player = player_cls()
        return player
    

    def reset_board(self):
        self._board = np.zeros((self.rows,self.columns))
        self._board[self.rows//2-1, self.columns//2-1] = -1
        self._board[self.rows//2-1, self.columns//2] = 1

        self._board[self.rows//2, self.columns//2-1] = 1
        self._board[self.rows//2, self.columns//2] = -1
        self.time_p1 = 0
        self.time_p2 = 0

    def check_has_move(self, board):
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                is_valid, _ = self.process_move((r_cnt, c_cnt), board)
                if is_valid == True:
                    return True    
        
        return False 

        
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
                    board[r+1:r+dist+1, c+1:c+dist+1] = tmp.copy()

                if dir == 'l':
                    tmp = board[r-dist:r, c-dist:c].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r-dist:r, c-dist:c] = tmp.copy()

        board = np.fliplr(board)
        c_tmp = board.shape[1]-1-c
        k = c_tmp-r 
        side_diag = np.diag(board, k)
        flanks = self.check_row(side_diag, min(r , c_tmp))
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r+1:r+dist+1, c_tmp+1:c_tmp+dist+1].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r+1:r+dist+1, c_tmp+1:c_tmp+dist+1] = tmp.copy()

                if dir == 'l':
                    tmp = board[r-dist:r, c_tmp-dist:c_tmp].copy()
                    np.fill_diagonal(tmp, 1)
                    board[r-dist:r, c_tmp-dist:c_tmp] = tmp.copy()

        board = np.fliplr(board)
        if moved == True:
            board[r, c] = 1   
            # if r== c==0:
            #     print(board, 'kkkkkkk', r, c)
            #     print()

        return moved, board
    
    
    def check_if_finished(self, board):
        if (board == 0).sum() == 0 or (board == 1).sum()==0 or (board == -1).sum()==0:
            return True 
        
        return False 


    def play(self, player1, player2):
        global timed_out
        timed_out = False 

        # Randomly swap p1, p2 for first move.
        if self.deterministic:
            p1, p1piece = (player1, +1)
            p2, p2piece = (player2, -1)
        else:
            toss = random.randint(0, 1)
            p1, p1piece = (player1, +1) if toss==1 else (player2, -1)
            p2, p2piece = (player2, -1) if toss==1 else (player1, +1)


        self.reset_board()
        winner, reason, moves = None, '', []
        
        p1_cls = self.load_players(p1)
        p2_cls = self.load_players(p2)


        with time_limit(self.total_time_limit, 'sleep'):
            p1_cls.setup()
        
            
        if timed_out == True:
            winner, reason = p2, 'Setup timeout'
            return winner, reason, moves
         

        with time_limit(self.total_time_limit, 'sleep'):
            p2_cls.setup()
        
        if timed_out == True:
            winner, reason = p1, 'Setup timeout'
            return winner, reason, moves
        
        
        p1_invalid, p2_invalid = 0, 0
        while True:

            p1_board = self._board * p1piece
        

            with time_limit(self.total_time_limit-self.time_p1, 'sleep'):
                t0 = time.time()
                move = p1_cls.play(p1_board.copy())
                t1 = time.time()
                self.time_p1 += t1-t0

                if self.time_p1 > self.total_time_limit:
                    timed_out = True
                    winner, reason = p2, 'Total Time limit exceeded'
                    break


                has_move = self.check_has_move(p1_board.copy())
                is_valid = True
                if move != None:
                    is_valid, p1_board = self.process_move(move, p1_board.copy())
                
                if has_move == True:
                    if move == None:
                        p1_invalid += 1
                        if p1_invalid >= self.max_invalid_moves:
                            winner, reason = p2, 'Invalid moves exceeded %d' % self.max_invalid_moves
                            break 

                if has_move == False:
                    if move != None:
                        p1_invalid += 1
                        if p1_invalid >= self.max_invalid_moves:
                            winner, reason = p2, 'Invalid moves exceeded %d' % self.max_invalid_moves
                            break 

                if  is_valid == False:
                    p1_invalid += 1
                    if p1_invalid >= self.max_invalid_moves:
                        winner, reason = p2, 'Invalid moves exceeded %d' % self.max_invalid_moves
                        break 
                else:
                    self._board = p1_board * p1piece
                    moves.append(move)
                    # print(self._board, move, 'p1')
                    # print()

                if self.check_if_finished(p1_board):
                    if (p1_board == 1).sum() > (p1_board == -1).sum():
                        winner, reason = p1, 'Majority'
                        break
                    elif (p1_board == 1).sum() < (p1_board == -1).sum():
                        winner, reason = p2, 'Majority'
                        break
                    elif (p1_board == 1).sum()  == (p1_board == -1).sum():
                        winner, reason = None, 'Game drawn'
                        break 
            
            
            if timed_out == True:
                winner, reason = p2, 'Move timeout'
                break 

            p2_board = self._board * p2piece
 
            with time_limit(self.total_time_limit-self.time_p2, 'sleep'):
                t0 = time.time()
                move = p2_cls.play(p2_board.copy())
                t1 = time.time()
                self.time_p2 += t1-t0

                if self.time_p2 > self.total_time_limit:
                    timed_out = True
                    winner, reason = p1, 'Total Time limit exceeded'
                    break

                has_move_2 = self.check_has_move(p2_board.copy())
                is_valid = True
            
                if move != None:
                    is_valid, p2_board = self.process_move(move, p2_board)
                    

                if has_move_2 == True:
                    if move == None:
                        p2_invalid += 1
                        if p2_invalid >= self.max_invalid_moves:
                            winner, reason = p1, 'Invalid moves exceeded %d' % self.max_invalid_moves
                            break 

                if has_move_2 == False:
                    if move != None:
                        p2_invalid += 1
                        if p2_invalid >= self.max_invalid_moves:
                            winner, reason = p1, 'Invalid moves exceeded %d' % self.max_invalid_moves
                            break 

                if  is_valid == False:
                    p2_invalid += 1
                    if p2_invalid >= self.max_invalid_moves:
                        winner, reason = p1, 'Invalid moves exceeded %d' % self.max_invalid_moves
                        break 
                else:
                    self._board = p2_board * p2piece
                    moves.append(move)
                    # print(self._board, move, 'p2', 'dfdfdf')
                    # print()

                if self.check_if_finished(p2_board) or (has_move == False and has_move_2 == False):
                    if (p2_board == 1).sum() > (p2_board == -1).sum():
                        winner, reason = p2, 'Majority'
                        # print('dfdfdfdfdf', has_move, has_move_2, p2_board)
                        break
                    elif (p2_board == 1).sum() < (p2_board == -1).sum():
                        winner, reason = p1, 'Majority'
                        break
                    elif (p2_board == 1).sum()  == (p2_board == -1).sum():
                        winner, reason = None, 'Game drawn'
                        break
        
            
            if timed_out == True:
                winner, reason = p1, 'Move timeout'
                break 
        
        return winner, reason, moves


    def play_multiple(
        self, player1: str, player2: str, num: int=1, alternate: bool=True, verbose: bool=False
    ) -> Dict[str, int]:
        record = {
            player1: 0, player2: 0, None: 0, 'moves':[]
        }

        for i in range(num):
            if i % 2 == 0:
                p1, p2 = player1, player2
            else:
                p1, p2 = player2, player1
            winner, reason, moves = self.play(p1, p2)
            record[winner] += 1
            record['moves'].append(((p1,p2), moves))
            if verbose:
                print('Winner: %s.\t%s' % (winner, reason))
        return record

    @property
    def board(self,):
        return self._board


def championship(
        arena_path: Iterable[str], game_options: Dict=None, num: int=1, verbose: bool=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    # If arena is a list with a single string, it is assumed to be the path to a directory.
    # The directory should contain python modules or packages. From each of them,
    # a class Player should be able to be imported with the step() and play()
    # methods.
    idx_insertion = None

    
    
    if len(arena_path)==1:
        idx_insertion = len(sys.path)
        abs_path = os.path.abspath(arena_path[0])
        sys.path.append(abs_path)
        arena = os.listdir(arena_path[0])
        arena = [
            # Convert .py files to module names, otherwise assume they are packages
            # arena_path[0] + '/' + p[:-3] + '/' + 'Player' if p.endswith('.py') else p \
            p[:-3] if p.endswith('.py') else p \
                
            for p in arena if (
                # Only look at files if they don't start with _, . (like __init)
                not (p.startswith('.') or p.startswith('_')) and \
                (p.endswith('.py') or os.path.isdir(os.path.join(abs_path, p)))
            )
        ]
    print(arena, 'ffff')
    # Victories is a 2D array, with a row for each player, and each column
    # containing the number of wins against each player. So victories[2,3] will
    # have the number of wins player 2 had over player 3.
    # The dictionary idx_ref maps a player name to its index in `victories`
    # So victories[idx_ref['name']] will return the row of victories for player
    # `name`.
    victories = np.zeros((len(arena), len(arena)), dtype=int)
    losses = np.zeros((len(arena), len(arena)), dtype=int)
    draws = np.zeros((len(arena), len(arena)), dtype=int)
    avg_moves = np.zeros_like(victories)
    idx_ref = {player_name:i for i, player_name in enumerate(arena)}
    # If arena was a string, it is now expanded into a list of player names with
    # the default Player class. If it already was a list of strings, those can
    # contain non-default class names like module.submodule/classname.
    # Now generating pairings of players for a game:
    max_len = max(map(len, arena))

    
    for player1, player2 in (combinations(arena, 2)):
        if verbose:
            print(f'{player1:>{max_len}} vs {player2:<{max_len}}')
        game = OthelloBoard(**game_options)

        record = game.play_multiple(player1, player2, num, alternate=True, verbose=verbose)
        if verbose:
            print('Moves:')
            for players, moves in record['moves']:
                print('\t%s:%s' % (players, moves))
        victories[idx_ref[player1], idx_ref[player2]] += record[player1]
        victories[idx_ref[player2], idx_ref[player1]] += record[player2]
        draws[idx_ref[player1], idx_ref[player2]] += record[None]
        draws[idx_ref[player2], idx_ref[player1]] += record[None]
        avg_moves[idx_ref[player1], idx_ref[player2]] = sum(len(m[1]) for m in record['moves']) / len(record['moves'])
        avg_moves[idx_ref[player2], idx_ref[player1]] = avg_moves[idx_ref[player1], idx_ref[player2]]
        if verbose:
            print(f'{record[player1]:>{max_len}} -- {record[player2]:<{max_len}}')
            if record[None] > 0: # draws
                print(f'Draws: {record[None]:^{2*max_len+4}}')
    losses = victories.T
    if idx_insertion is not None:
        del sys.path[idx_insertion] # leave sys.path unchanged after function returns
    return victories, losses, draws, avg_moves, idx_ref


    
if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Othello  Game',
        description='Play Othello between two players.')
    parser.add_argument(
        '-v', '--versus', nargs=2, metavar=('P1', 'P2'),
        help=('Play one game between two players. Specify players as '
              '`MODULE.PATH/CLASSNAME` or `MODULE.PATH` where the default `Player` '
              'class is used. For e.g. `dummy/LazyBoi`, or `dummy` (which will '
              'use the `dummy.Player` class.')
    )
    parser.add_argument(
        '-c', '--championship', nargs='+', metavar='DIRECTORY',
        help=('Specify directory containing player modules/packages, OR list '
              'of player modules/packages. Each player plays against every other '
              'player. If directory given, each module/package should implement '
              'the default `Player` class.')
    )
    parser.add_argument(
        '-n', '--num', type=int, default=1,
        help='Number of games to play for a pair in a championship.'
    )
    parser.add_argument(
        '--rows', type=int, default=ROWS,
        help='Number of rows in game board.'
    )
    parser.add_argument(
        '--columns', type=int, default=COLUMNS,
        help='Number of columns in game board.'
    )
    parser.add_argument(
        '--timeout_move', type=float, default=TIMEOUT_MOVE,
        help='Time alotted per player move.'
    )
    parser.add_argument(
        '--timeout_setup', type=float, default=TIMEOUT_SETUP,
        help='Time alotted for setup before each game.'
    )
    parser.add_argument(
        '--max_invalid_moves', type=int, default=MAX_INVALID_MOVES,
        help='Max invalid moves before forfeiting the game.'
    )
    parser.add_argument(
        '--suppress', default=False, action='store_true',
        help='Whether to suppress stdout of player processes.'
    )
    args = parser.parse_args()

    if args.versus is not None and args.championship is not None:
        print('Only specify either `versus` or `championship` option.', file=sys.stderr)
        exit(-1)

    game_options = dict(
        rows=args.rows, columns=args.columns,
        timeout_move=args.timeout_move, timeout_setup=args.timeout_setup,
        max_invalid_moves=args.max_invalid_moves,
        suppress=args.suppress
    )

    if args.versus is not None:
        game = OthelloBoard(**game_options)
        winner, reason, moves = game.play(args.versus[0], args.versus[1])
        print('Winner %s, due to %s' % (winner, reason))
        print(game)
    else:
        vic, los, dra, mov, ref = championship(args.championship, game_options, args.num)
        reverse_ref = {idx: name for name, idx in ref.items()}
        max_len = max(map(len, ref.keys()))
        totals = np.sum(vic, axis=1)
        total_draws = np.sum(dra, axis=1)
        total_loss = np.sum(los, axis=1)
        total_moves = np.mean(mov, axis=1)
        rankings = np.argsort(totals)[::-1]
        print('{:{max_len}s}\t{:2s}\t{:2s}\t{:2s}\t{:2s}'.format('Player', 'Wins', 'Draws', 'Losses', 'Moves', max_len=max_len))
        for idx in rankings:
            print('{:{max_len}s}\t{:2d}\t{:2d}\t{:2d}\t{:2f}'.format(reverse_ref[idx], totals[idx], total_draws[idx], total_loss[idx], total_moves[idx], max_len=max_len))
    exit(0)
