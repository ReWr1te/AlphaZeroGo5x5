import random
import numpy as np
from collections import defaultdict, deque
from go import GO
from cnn_keras import PolicyValueNet
from my_player import MyPlayer
from random_player import RandomPlayer
from greedy_player import GreedyPlayer
from aggressive_player import AggressivePlayer
from smart_player import SmartPlayer


class Train(object):
    """ Training Pipeline of Policy Value Neural Network. """

    def __init__(self, saved_weights=None):
        """ Initialize Attributes. """
        # board attributes
        self.size = 5  # board (go) size
        self.go = GO(self.size)  # initialize the board (go)
        # training params
        # mine, adjust this manually to set training process
        # --------------------------------------------------------------------
        self.R = 200  # num of simulations (rollouts) for each move
        self.check_freq = 50  # the frequency to check performance
        self.game_batch_num = 5000  # total batch number of selfplays
        self.test_num = 50  # test games number
        # --------------------------------------------------------------------
        # github preset, do not change
        self.lr = 2e-3  # learning rate of the whole process
        self.lr_coef = 1.0  # adaptively adjust lr based on KL
        self.temp = 1.0  # the temperature param controlling exploration
        self.C = 5  # hyperparameter controls the weight of prior probs
        self.buffer_size = 10000  # buffer size for data retrieving
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)  # data buffer
        self.play_batch_size = 1  # batch size for playing each time
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02  # kl target
        self.best_win_ratio = 0.0  # best_win_ratio to compare models
        # set policy-value net
        self.pv_net = PolicyValueNet(self.size, saved_weights)
        # set my player
        self.mcts_player = MyPlayer(
            self.pv_net.policy, c=self.C, r=self.R,
            is_selfplay=True, is_train=True)
        # set opponent players to evaluate performance
        self.op_player_n = 0
        self.op_players = [
            RandomPlayer(), GreedyPlayer(), AggressivePlayer(), SmartPlayer()]

    def self_play(self, player, verbose=False, temp=1e-3):
        """ Self-play using MCTS player.
        Params:
            player: MCTS player object.
            verbose: bool, show board or not.
            temp: float, controls the weight of exploration.
        Returns: tuple of winner and game data.
        """
        self.go = GO(self.size)
        states, mcts_probs, players = [], [], []
        if verbose:
            self.go.visualize_board()
        while True:
            piece_type = 1 if self.go.X_move else 2
            if self.go.game_end(piece_type):
                # winner from the perspective of the current player
                winner = self.go.judge_winner()
                winners = np.zeros(len(players))
                winners[np.array(players) == winner] = 1.0
                winners[np.array(players) != winner] = -1.0
                # reset MCTS root node
                player.reset()
                if verbose:
                    print('Game ended.')
                    str_winner = 'X' if winner == 1 else 'O'
                    print(f'The winner is {str_winner}')
                return winner, zip(states, mcts_probs, winners)
            move, move_probs = player.get_move(
                self.go, temp=temp, return_prob=1)
            # store the data
            states.append(self.pv_net.get_state(self.go))
            mcts_probs.append(move_probs)
            players.append(piece_type)
            # perform a move
            i, j = move // self.go.size, move % self.go.size
            if not self.go.place_chess(i, j, piece_type):
                if verbose:
                    self.go.visualize_board()
                continue
            self.go.died_pieces = self.go.remove_died_pieces(3 - piece_type)
            self.go.X_move = not self.go.X_move

    def get_aug_data(self, data):
        """ Augment data by rotating and flipping.
        Params: data: list of (state, probs, winner) tuples.
        Returns: aug_data: list of 4 augmented data of the original one.
        """
        aug_data = []
        for state, mcts_porb, winner in data:
            for i in range(1, 5):
                # rotate
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_porb.reshape(self.size, self.size)), i)
                aug_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                aug_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return aug_data

    def buffer_selfplay_data(self, n_games=1):
        """ Buffer selfplay data.
        Params: n_games: int, number of selfplay games.
        """
        for i in range(n_games):
            winner, data = self.self_play(self.mcts_player, temp=self.temp)
            data = list(data)[:]
            self.last_moves = len(data)
            data = self.get_aug_data(data)
            self.data_buffer.extend(data)

    def update_weights(self):
        """ Update neural network weights using training data.
        Returns: float loss and entropy.
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        s_batch = [data[0] for data in mini_batch]
        mp_batch = [data[1] for data in mini_batch]
        w_batch = [data[2] for data in mini_batch]
        old_probs, old_val = self.pv_net.model.predict_on_batch(
            np.array(s_batch))
        for i in range(self.epochs):
            loss, entropy = self.pv_net.train_core(
                s_batch, mp_batch,
                w_batch, self.lr*self.lr_coef)
            new_probs, new_val = self.pv_net.model.predict_on_batch(
                np.array(s_batch))
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_coef > 0.1:
            self.lr_coef /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_coef < 10:
            self.lr_coef *= 1.5
        explained_var_old = (
            1 - np.var(np.array(w_batch) - old_val.flatten()) /
            np.var(np.array(w_batch)))
        explained_var_new = (
            1 - np.var(np.array(w_batch) - new_val.flatten()) /
            np.var(np.array(w_batch)))
        print(("kl:{:.5f}, "
               "lr_coef:{:.3f}, "
               "loss:{}, "
               "entropy:{}, "
               "explained_var_old:{:.3f}, "
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_coef,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=50):
        """ Evaluate the neural network policy.
        Params: n_games: int, number of games will be played.
        Returns: win_ratio: float, winning ratio of this evaluation.
        """
        p1 = MyPlayer(self.pv_net.policy, c=self.C, r=self.R, is_train=True)
        # p1 = MyPlayer(c=self.C, r=self.R)
        p2 = self.op_players[self.op_player_n]
        wins = defaultdict(int)
        for i in range(n_games):
            self.go = GO(self.size)
            if i % 2 == 0:
                winner = self.go.play(p1, p2, verbose=False)
                wins[p1.type if winner == 1 else p2.type] += 1
            else:
                winner = self.go.play(p2, p1, verbose=False)
                wins[p1.type if winner == 2 else p2.type] += 1
        win_ratio = 1.0 * wins[p1.type] / n_games
        print(" op_player: {}, win: {}, lose: {}, win_ratio:{}".format(
            p2.type, wins[p1.type], wins[p2.type], win_ratio))
        return win_ratio

    def train(self):
        """ Training process. """
        try:
            for i in range(self.game_batch_num):
                self.buffer_selfplay_data(self.play_batch_size)
                print("Batch No.{} Moves:{}".format(
                        i+1, self.last_moves))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.update_weights()
                # check performance and save weights
                if (i+1) % self.check_freq == 0:
                    print(f" Self-play batch: {i+1}. Games: {self.test_num}")
                    win_ratio = self.policy_evaluate(self.test_num)
                    self.pv_net.save_weights('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print(" New best policy weight. Saving... ")
                        self.best_win_ratio = win_ratio
                        # save best policy weights
                        self.pv_net.save_weights('./best_policy.model')
                        if self.best_win_ratio >= 0.98:
                            if self.op_player_n != 3:
                                self.op_player_n += 1
                                self.best_win_ratio = 0.0
                            else:
                                if self.test_num >= 100:
                                    break
                                self.test_num += 10
                                self.best_win_ratio = 0.95
        except KeyboardInterrupt:
            print('\n\rExiting...')


if __name__ == '__main__':
    pipeline = Train()
    pipeline.train()
