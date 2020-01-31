import numpy as np
from copy import deepcopy


def softmax(x):
    """ Return adjusted softmax value of x. """
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


class Node(object):
    """ Monte Carlo Tree Node. """

    def __init__(self, parent, p):
        """ Initialize Attributes. """
        self.parent = parent  # parent node object
        self.children = {}  # child node object
        self.N = 0  # int, number of visits
        self.P = p  # float, prior probability
        self.Q = 0  # float, Q value

    def is_leaf(self):
        """ Return whether the node is a leaf node. """
        return self.children == {}

    def is_root(self):
        """ Return whether the node is a root node. """
        return self.parent is None

    def uct(self, c):
        """ Get UCT score of this node.
        Params: c: float, hyperparameter to control the weight of prior probs.
        Returns: float, UCT score of this node.
        """
        return self.Q + c * self.P * np.sqrt(self.parent.N) / (1 + self.N)

    def select(self, c):
        """ Select move and corresponding child node in children.
        Params: c: float, hyperparameter to control the weight of prior probs.
        Returns: tuple, (move, child node).
        """
        return max(self.children.items(), key=lambda m_c: m_c[1].uct(c))

    def expand(self, move_porbs):
        """ Expand MCT.
        Params: move_probs: list of (move, probability) tuples.
        """
        for m, p in move_porbs:
            if m not in self.children:
                self.children[m] = Node(self, p)

    def backup(self, val):
        """ Back up the value from leaves to root.
        Params: val: float, value from a leaf.
        """
        self.N += 1
        self.Q += (val - self.Q) / self.N
        if self.parent:
            self.parent.backup(-val)  # parent is a different player


class MCTS(object):
    """ Monte Carlo Tree Search Class. """
    def __init__(self, policy, c=5, r=200):
        """ Initilize Attributes. """
        self.root = Node(None, 1.0)  # root node with prior probability 1.0
        self.policy = policy  # policy function from CNN
        self.C = c  # float weight controling prior probs
        self.R = r  # number of rollouts

    def rollout(self, go):
        """ Rollout to get simulated game result and back it up.
        Params: go: must be a copy of a go object.
        """
        cur_node = self.root
        while True:
            piece_type = 1 if go.X_move else 2
            if cur_node.is_leaf():  # down to the leaf node
                break
            move, cur_node = cur_node.select(self.C)
            if not go.place_chess(move // go.size, move % go.size, piece_type):
                continue
            go.died_pieces = go.remove_died_pieces(3 - piece_type)
            go.X_move = not go.X_move
        move_porbs, val = self.policy(go)
        # if not the end, expand this node
        if not go.game_end(piece_type):
            cur_node.expand(move_porbs)
        else:
            winner = go.judge_winner()
            case1 = (winner == 1) and go.X_move
            case2 = (winner == 2) and not go.X_move
            val = 1.0 if (case1 or case2) else -1.0
        cur_node.backup(-val)

    def get_move_probs(self, go, temp=1e-3):
        """ Roll out and get (move, probs) tuples.
        Params:
            go: a go object.
            temp: float, hyperparameter controling the degree of exploration
        Returns: tuple, (moves, move_probs)
        """
        for _ in range(self.R):
            self.rollout(deepcopy(go))  # must copy the board of go
        # get moves and corresponding visit numbers to calculate probabilities
        moves, visits = zip(*[(m, c.N) for m, c in self.root.children.items()])
        return moves, softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

    def update_move(self, last_move):
        """ Move forward to the leaves.
        Params: last_move: int, last move numeber.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
