import chess.pgn
import torch
import torch.nn.functional as F


class ChessDatasetV0:

    def __init__(self, n_train=1_000_000, n_valid=0, n_test=0):
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.valid = {}
        self.test = {}

        self.FILE = "/home/severino/Workspace/projects/road_to_transformer/datasets/chess/lichess_db_standard_rated_2024-06.pgn"
        self.COLUMNS = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

        games = open(self.FILE)

        print("[chess.py] Preparing valid.")
        while len(self.valid) < self.n_train:
            game = chess.pgn.read_game(games)
            if game is None:
                raise Exception("[chess.py] No games left for completing valid.")
            game_id, game_moves = self.extract_game(game)
            if game_id not in self.valid:
                self.valid[game_id] = game_moves


    def extract_game(self, game):
        game_id = ""
        game_moves = []
        for i, move in enumerate(game.mainline_moves()):
            assert len(move) == 4, f"[chess.py] Movement \"{move}\" does not have length 4 as expected."
            game_id += str(move)
            player_id = torch.tensor([i % 2])  # 0 -> White Player, 1 -> Black Player
            from_column = F.one_hot(torch.tensor(self.COLUMNS[move[0]]))
            from_row = F.one_hot(torch.tensor(int(move[1]) - 1))
            to_column = F.one_hot(torch.tensor(self.COLUMNS[move[2]]))
            to_row = F.one_hot(torch.tensor(int(move[3]) - 1))
            # TODO: maybe add elo
            move_tensor = torch.cat((player_id, from_column, from_row, to_column, to_row), dim=1)
            game_moves.append(move_tensor)
        return game_id, game_moves

    def train_iterator(self):
        raise NotImplementedError