import torch

Game = {
    'boardsize' : (6, 7),
    'actionsize' : 7,
}

Args = {
    'lr': 0.2,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
    'maxlen': 50000,
    'numGamesPerGen': 1,
    'mcts': 400,
    'mcts_alg': "mcts_par_mcts_root_par",
    'num_workers': 8,
    'c_puct': 0.10,
    'alpha': 0.3,
    'eps': 0.25,
}

"""
mcts_alg can be chosen from: mcts_mcts, mcts_par_mcts_notpar, mcts_par_mcts_par, mcts_par_mcts_root_par
For details, see rust_part/src/duel.rs
"""