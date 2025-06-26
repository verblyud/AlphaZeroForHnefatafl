from .NeuralNet import NNetWrapper, load_wrapper
from .taflNNet import TaflNNet
from .utils import *
from .config import Game, Args

# whatever code interpreter you are using might give a warning, but shouldn't matter.
# self note: I'm not sure why, but calling
# from ._azhnefatafl import self_play_function or
# from azhnefatafl._azhnefatafl import self_play_function 
# will fail.  30.01.2025 Keigo

from ._azhnefatafl import con4_self_play_function
