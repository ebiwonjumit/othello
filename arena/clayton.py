'''
This is the agent for the static board score heuristic, where different squares get different values based on utility.

Iterative deepening IS used for this agent.

This agent employs a simple static weighting heuristic, c/o @hylbyj on GitHub:
    https://github.com/hylbyj/Alpha-Beta-Pruning-for-Othello-Game/blob/master/readme_alpha_beta.txt

The paper where @hylbyj found the heuristic is by Sannidhanam and Annamalai at U of Washington, and it can be found here: 
    https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
'''

import numpy as np
import time

rows, cols = 8, 8

# Agent class for agent WITH heuristic AND iterative deepening
class Player:
    def __init__(self) -> None:
        super().__init__()

        self.name = "Clayton\'s Minimax Player"

        # Board square weights to calculte the heuristic
        self.weights = np.array(
            [[4, -3, 2, 2, 2, 2, -3, 4],
            [-3, -4, -1, -1, -1, -1, -4, -3],
            [2, -1, 1, 0, 0, 1, -1, 2],
            [2, -1, 0, 1, 1, 0, -1, 2],
            [2, -1, 0, 1, 1, 0, -1, 2],
            [2, -1, 1, 0, 0, 1, -1, 2],
            [-3, -4, -1, -1, -1, -1, -4, -3],
            [4, -3, 2, 2, 2, 2, -3, 4]]
        )

        print(self.__class__.__name__ + ': Clayton\'s Minimax Player.')

     # Setup method, overrides parent Player
    def setup(self):

        # Store rows and columns for reference
        self.rows = rows
        self.columns = cols

        # Track the execution time for iterative deepening 
        # During minimax alpha-beta pruning, the algorithm will continue to run (increasing its depth each time)
        #   until the "timeout_after_time" threshold is hit. At that point, the "play" method will return the best
        #   move from the deepest completed iteration

        self.start_time = 0

        # Absolute threshold for move is 1.0s, so the idea is to be as close as possible to 1.0 without EVER going over
        self.timeout_after_time = 1.25

        # Counter for total moves. Used for printing and debugging
        # Can be commented out for actual live play
        self.total_moves = 0

        # Track total time for cutoff
        self.total_time = 0


    # This value function multiplies the board by its weights (element-wise) and then computes the sum for the value function
    def get_board_value(self, board):
        return np.sum(np.multiply(board,self.weights))
    
    # This function returns the valid moves
    # Importantly, it not only returns moves, but also the corresponding boards,
    #   which is done to avoid repeated calls later on
    # valid_moves was adopted from the dummies.py file
    def valid_moves(self, board):
        moves = []

        # Boards returned as well with corresponding moves to avoid repeated calls later
        boards = []

        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):

                # We're already getting the board here when process_move is run
                # Instead of repeating this call later on to get a move's board, save the board here
                #   along with the move
                is_valid, b = self.process_move((r_cnt, c_cnt), board.copy())
                if is_valid == True:
                    moves.append((r_cnt, c_cnt))
                    boards.append(b)
        return moves, boards
        
    # This function takes a vector as input and calculates the flanks
    # For use during process_move
    # check_row was taken from the dummies.py file
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

    # This function processes a given move on a given board
    # It returns whether the new move was valid, along with the new board after the move
    # process_move was taken from the dummies.py file
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

        return moved, board
    
    # This is the play function required by player.py
    # The default functionality, unless overridden, employs iterative deepening
    # In it, calls are made to get the best move via minimax with alpha-beta pruning
    # Calls to minimax (get_best_move) are done via iterative deepening as long as allowed
    #   by the time constraint timeout_after_time
    def play(self, board: np.ndarray):

        # Track the start time of the move call for the iterative deepening time threshold
        self.start_time = time.time()

        # Define the initial max_depth of plies to start at for iterative deepening
        # This number will increment at every call to get_best_move until the timeout_after_time
        #   threshold is hit
        max_depth = 3

        # Store the best move from each call to get_best_move as we go deeper and deeper
        # Importantly, the initial max_depth value needs to NEVER time out, otherwise the play function
        #   will return None, and this would be an invalid move
        best_move = None

        # Run the iterative deepening minimax with alpha-beta pruning
        try:

            # Iteratively deepen until the timeout_after_time threshold is hit
            while True:

                # Store the best move each time minimax (get_best_move) is called
                _, best_move = self.get_best_move(board=board.copy(),
                                     depth=0,
                                     player=1,
                                     alpha=-np.inf,
                                     beta=np.inf,
                                     max_depth=max_depth,
                                     prev_no_action_possible=False) 
                

                # Increase max_depth for next iteration
                max_depth += 1

        # The exception is guaranteed to get hit, as the get_best_move will raise an exception based on the timer
        except:

            # While writing this up, I used this print statement to visualize moves, 
            #   but it will probably get commented out for actual game play
            
            # Increment total time
            self.total_time += round(time.time() - self.start_time,6)

            print(f"{self.name} move {self.total_moves} at {best_move}, max_depth={max_depth} at {round(time.time() - self.start_time,6)} seconds (total time {round(self.total_time,3)})")
                
        # The result of calling play is that the best move will be returned
        finally:

            # Increase total moves for player counter every time method is called
            self.total_moves += 1

            return best_move

    # minimax implementation with alpha-beta pruning (returns (value, move))
    # In addition to the standard minimax parameters, I also include a parameter that
    #   tracks whether or not the previous move was a non-action
    # This is because, if both players can't move, there is no reason to keep recursing because the game is over
    # The best action is used for returning to the play function, but the best value is used for recursive calls 
    def get_best_move(self, board, depth, player, alpha, beta, max_depth, prev_no_action_possible):

        # The fist check I do is my timer:
        #   when the timeout_after_time threshold is hit, an exception is immediately raised, and the 
        #   except block is hit in the play function to stop iterating and return a move
        if time.time() - self.start_time >= self.timeout_after_time:
            raise Exception(f"timeout_after_time threshold hit.")

        # This is the base case
        # Only the value is returned and backed up, as it doesn't matter 
        #   what the move was at any depth in the tree other than depth 0
        # If at max_depth, time to stop recursing and back up the values
        if depth == max_depth:

            # The value of the board will depend on which player is playing
            if player == 1:
                return self.get_board_value(board), None
            else:
                return self.get_board_value(board*(-1)), None

        # If not at max_depth, get all the moves...

        # Get valid moves and their boards from the current board
        moves, boards = self.valid_moves(board)

        # This tracks whether or not there are no actions to execute for the current player
        # If there are no current moves, it needs to be tracked
        curr_no_action_possible = False
        if len(moves) == 0:
            curr_no_action_possible = True

        # If there were consectutive non-actions for both players,
        #   there is no need to continue recursing on this branch, so the best value is returned
        if curr_no_action_possible and prev_no_action_possible:
            if player == 1:
                return self.get_board_value(board), None
            else:
                return self.get_board_value(board*(-1)), None

        # If the current player can't move, but the previous player could move, 
        #   the board is passed back to the other player and the current player passes its move
        if curr_no_action_possible:
            return self.get_best_move(board.copy()*(-1), depth+1, player*(-1), alpha, beta, max_depth, curr_no_action_possible)

        # Maximizing 
        if player == 1:

            # Track best values and their corresponding moves
            best_val = -np.inf
            best_move = None

            # Iterate through moves
            for i in range(len(moves)):

                # Recursive call, passing oppoent's board
                value, _ = self.get_best_move(boards[i].copy()*(-1),depth+1,player*(-1),alpha,beta,max_depth,curr_no_action_possible)

                # If the best_value improves (for the maximizing node), update it and its corresponding move
                if value > best_val:
                    best_val = value
                    best_move = moves[i]

                # Update alpha
                alpha = max(alpha,best_val)

                # alpha-beta pruning: break if beta is lte alpha, as there is no need to continue
                if beta <= alpha:
                    break
            
            # Return the best value and best move
            return best_val,best_move

        # Minimizing
        else:

            # Track best values and their corresponding moves
            best_val = np.inf
            best_move = None

            # Iterate through moves
            for i in range(len(moves)):

                # Recursive call, passing oppoent's board
                value, _ = self.get_best_move(boards[i].copy()*(-1),depth+1,player*(-1),alpha,beta,max_depth,curr_no_action_possible)

                # If the best_value improves (for the minimizing node), update it and its corresponding move
                if value < best_val:
                    best_val = value
                    best_move = moves[i]

                # Update beta
                beta = min(beta,best_val)

                # alpha-beta pruning: break if beta is lte alpha, as there is no need to continue
                if beta <= alpha:
                    break

            # Return the best value and best move
            return best_val,best_move

'''
python othello.py -v manda_mcts/MCTSAgent clayton/Player
python othello.py -v clayton/Player manda_mcts/MCTSAgent
'''