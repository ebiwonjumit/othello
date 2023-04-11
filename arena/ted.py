import numpy as np
from typing import Tuple



class Player():

    def __init__(self, max_depth: int = 4, time_limit: float = 5.0):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.player = 1
        self.opponent = -1
        self.invalid_moves = 0

    def setup(self):
        self.rows = 8
        self.columns = 8
        self.player = 1
        self.opponent = -1
        self.invalid_moves = 0
        print('Ted Alpha Beta')

    def play(self, board: np.ndarray) -> Tuple[int, int]:
        self.invalid_moves = 0
        self.player = 1
        self.opponent = -1
        move = self.alpha_beta_search(board)
        return move

    def alpha_beta_search(self, board: np.ndarray) -> Tuple[int, int]:
        alpha = float('-inf')
        beta = float('inf')
        best_move = None
        for depth in range(1, self.max_depth + 1):
            value, move = self.max_value(board, depth, alpha, beta)
            if value == float('inf'):
                break
            best_move = move

        return best_move

    def max_value(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> Tuple[float, Tuple[int, int]]:
        if depth == 0 :
            return self.heuristic(board), None
        moves = self.valid_moves(board, self.player)
        if not moves:
            return self.heuristic(board), None
        value = float('-inf')
        best_move = None
        for move in moves:
            new_board = self.apply_move(board, move, self.player)
            new_value, _ = self.min_value(new_board, depth - 1, alpha, beta)
            if new_value > value:
                value = new_value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move

    def min_value(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> Tuple[float, Tuple[int, int]]:
        if depth == 0 :
            return self.heuristic(board), None
        moves = self.valid_moves(board, self.opponent)
        if not moves:
            return self.heuristic(board), None
        value = float('inf')
        best_move = None
        for move in moves:
            new_board = self.apply_move(board, move, self.opponent)
            new_value, _ = self.max_value(new_board, depth - 1, alpha, beta)
            if new_value < value:
                value = new_value
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move

    def valid_moves(self, board: np.ndarray, player: int) -> list:
        moves = []
        for r_cnt in range(self.rows):
            for c_cnt in range(self.columns):
                is_valid, _ = self.process_move((r_cnt, c_cnt), board.copy(), player)
                if is_valid:
                    moves.append((r_cnt, c_cnt))
        return moves

    def process_move(self, move, board, player):
        r, c = move
        if board[r, c] != 0:
            return False, board

        flanks = self.check_row(board[r, :], c, player)
        moved = False
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    board[r, c + 1:c + dist + 1] = player

                if dir == 'l':
                    board[r, c - dist:c] = player

        flanks = self.check_row(board[:, c].T, r, player)
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    board[r + 1:r + 1 + dist, c] = player

                if dir == 'l':
                    board[r - dist:r, c] = player

        k = c - r
        main_diag = np.diag(board, k)
        flanks = self.check_row(main_diag, min(r, c), player)
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r + 1:r + dist + 1, c + 1:c + dist + 1].copy()
                    np.fill_diagonal(tmp, player)
                    board[r + 1:r + dist + 1, c + 1:c + dist + 1] = tmp

                if dir == 'l':
                    tmp = board[r - dist:r, c - dist:c].copy()
                    np.fill_diagonal(tmp, player)
                    board[r - dist:r, c - dist:c] = tmp

        board = np.fliplr(board)
        k = board.shape[1] - 1 - c - r
        side_diag = np.diag(board, k)
        flanks = self.check_row(side_diag, min(r, board.shape[1] - 1 - c), player)
        if len(flanks) > 0:
            moved = True
            for dir, dist in flanks:
                if dir == 'r':
                    tmp = board[r + 1:r + dist + 1, c + 1:c + dist + 1].copy()
                    np.fill_diagonal(tmp, player)
                    board[r + 1:r + dist + 1, c + 1:c + dist + 1] = tmp

                if dir == 'l':
                    tmp = board[r - dist:r, c - dist:c].copy()
                    np.fill_diagonal(tmp, player)
                    board[r - dist:r, c - dist:c] = tmp

        board = np.fliplr(board)
        if moved == True:
            board[r, c] = player

        return moved, board

    def check_row(self, vec, c, player):
        c_num = vec.size
        flanks = []
        if (vec == player).sum() > 0:
            idxs = np.nonzero(vec == player)[0]
            dists = c - idxs
            if c < c_num - 1 and (dists < 0).sum() > 0:
                min_dist_right = np.abs(dists[dists < 0]).min()
                if np.all(vec[c + 1:c + min_dist_right] == -player) and min_dist_right > 1:
                    flanks.append(('r', min_dist_right - 1))

            if c > 0 and (dists > 0).sum() > 0:
                min_dist_left = np.abs(dists[dists > 0]).min()
                if np.all(vec[c - min_dist_left + 1:c] == -player) and min_dist_left > 1:
                    flanks.append(('l', min_dist_left - 1))

        return flanks

    def apply_move(self, board: np.ndarray, move: Tuple[int, int], player: int) -> np.ndarray:
        is_valid, new_board = self.process_move(move, board.copy(), player)
        if is_valid:
            return new_board
        else:
            raise ValueError(f"Invalid move: {move}")

    def heuristic(self, board: np.ndarray) -> int:
        player_pieces = np.sum(board == self.player)
        opponent_pieces = np.sum(board == -self.player)
        return player_pieces - opponent_pieces