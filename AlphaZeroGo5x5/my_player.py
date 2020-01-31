import numpy as np
from cnn_keras import PolicyValueNet
from mcts import MCTS


class MyPlayer(object):
    """ Monte Carlo Tree Search Player. """

    def __init__(
            self, policy=None, c=5, r=100, is_selfplay=False, is_train=False):
        """ Initialize attributes. """
        self.type = 'my'   # symbol for go_play.py
        if not policy:
            if is_train:
                policy = PolicyValueNet(5, 'current_policy.model').policy
            else:
                policy = PolicyValueNet(5, 'best_policy.model').policy
        self.mcts = MCTS(policy, c, r)  # MCTS class
        self.is_selfplay = is_selfplay  # whether selfplay

    def reset(self):
        """ Reset MCTS to the root node. """
        self.mcts.update_move(-1)

    def get_candidates(self, go):
        """ Get available move options for current board.
        Params: go: a go object.
        Returns: candidates: list of int, available moves.
        """
        piece_type = 1 if go.X_move else 2
        candidates = []
        for i in range(go.size**2):
            row, col = i // go.size, i % go.size
            if go.valid_place_check(row, col, piece_type):
                candidates.append(i)
        return candidates

    def get_move(self, go, temp=1e-3, return_prob=False):
        """ Get move by MCTS.
        Params:
            go: a go object.
            temp: float, controls the weight of exploration.
            return_prob: bool, whether return the probs of moves
        Returns: (move, move_probs) if return_prob else move)
        """
        candidates = self.get_candidates(go)
        move_probs = np.zeros(go.size**2)
        if len(candidates) > 0:
            moves, probs = self.mcts.get_move_probs(go, temp)
            move_probs[list(moves)] = probs
            if self.is_selfplay:
                noise = np.random.dirichlet(0.3 * np.ones(len(probs)))
                move = np.random.choice(moves, p=0.75*probs+0.25*noise)
                self.mcts.update_move(move)
            else:
                move = np.random.choice(moves, p=probs)
                self.mcts.update_move(-1)
            return (move, move_probs) if return_prob else move
        else:
            print("No available moves. ")

    def get_input(self, go, piece_type):
        """ go_play.py encapsulation of get_move(). """
        move = self.get_move(go)
        return move // go.size, move % go.size
