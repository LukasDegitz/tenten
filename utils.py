import numpy as np
import random as rd
from collections import namedtuple

pieces = [
    np.array([[1]]),
    np.array([[1, 1]]),
    np.array([[1, 1, 1]]),
    np.array([[1, 1, 1, 1]]),
    np.array([[1, 1, 1, 1, 1]]),
    np.array([[1], [1]]),
    np.array([[1], [1], [1]]),
    np.array([[1], [1], [1], [1]]),
    np.array([[1], [1], [1], [1], [1]]),
    np.array([[0, 1], [1, 1]]),
    np.array([[1, 0], [1, 1]]),
    np.array([[1, 1], [0, 1]]),
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]]),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
]

# An action consists in three components: an id referencing a piece (p) and the target position i,j on the board (b)
Action = namedtuple("Action", ["p_id", "b_i", "b_j"])

def get_pieces():
    return rd.choices(pieces, k=3)

def little_gauss(n):
    #Bester Mann!!
    return int((n*n + n)/2)
